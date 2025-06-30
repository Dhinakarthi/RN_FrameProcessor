import React from 'react';
import { NavigationContainer, useNavigation } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { Text, TouchableOpacity, View } from "react-native";
import CameraScreen from "./CameraScreen";
import { launchImageLibrary } from 'react-native-image-picker';
import ImageResizer from '@bam.tech/react-native-image-resizer';
import * as rgb from 'react-native-image-to-rgb';
import { useTensorflowModel } from "react-native-fast-tflite";
import MainScreen from './MainScreen';
import TestScreen from './TestScreen';

const App = () => {
  function HomeScreen() {
    const navigation = useNavigation();

    const detector = useTensorflowModel(require('./assets/EasyOCR_EasyOCRDetector.tflite'));
    const recognizer = useTensorflowModel(require('./assets/EasyOCR_EasyOCRRecognizer.tflite'));
    const alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,- /_".split('');

    const runDetector = async (imageUri) => {
      console.log("🔍 Starting detector...");

      const [, , detH, detW] = detector.model.inputs[0].shape;
      console.log(`🖼️ Detector input size: ${detW} x ${detH}`);

      const resized = await ImageResizer.createResizedImage(imageUri, detW, detH, 'JPEG', 100, 0);
      console.log("📏 Resized image URI:", resized.uri);

      const rgbArray = await rgb.convertToRGB(resized.uri);
      console.log("🎨 Raw RGB array length:", rgbArray.length);

      // Safe normalization with diagnostics
      const inputTensor = new Float32Array(rgbArray.length);
      let nonZeroCount = 0;
      for (let i = 0; i < rgbArray.length; i++) {
        const val = rgbArray[i];
        const norm = Math.min(1, Math.max(0, val / 255));
        inputTensor[i] = norm;
        if (val > 5) nonZeroCount++;
      }

      console.log(`📊 Normalized tensor shape: [1, 3, ${detH}, ${detW}]`);
      console.log("📊 Sample normalized values:", Object.fromEntries(inputTensor.slice(0, 6).entries()));
      console.log(`🧪 Non-zero RGB count: ${nonZeroCount} / ${rgbArray.length}`);

      // Run detector
      const output = await detector.model.run([inputTensor]);
      const outputArr = Array.isArray(output) ? output[0] : Object.values(output)[0];
      const [_, outH, outW, outC] = detector.model.outputs[0].shape;

      console.log("📤 Detector output shape:", [outH, outW, outC]);
      console.log("📤 Output tensor length:", outputArr.length);

      // Thresholds – Adjust for debug
      const TEXT_THRESHOLD = 0.5;
      const LINK_THRESHOLD = 0.5;

      const textMask = [], linkMask = [];
      let textTrueCount = 0, linkTrueCount = 0;
      let scoreMin = Infinity, scoreMax = -Infinity;
      let linkMin = Infinity, linkMax = -Infinity;

      for (let i = 0; i < outputArr.length; i += outC) {
        const rawScore = outputArr[i];
        const rawLink = outputArr[i + 1];

        const score = 1 / (1 + Math.exp(-rawScore)); // sigmoid
        const link = 1 / (1 + Math.exp(-rawLink));

        scoreMin = Math.min(scoreMin, score);
        scoreMax = Math.max(scoreMax, score);
        linkMin = Math.min(linkMin, link);
        linkMax = Math.max(linkMax, link);

        const textVal = score > TEXT_THRESHOLD;
        const linkVal = link > LINK_THRESHOLD;

        if (textVal) textTrueCount++;
        if (linkVal) linkTrueCount++;

        textMask.push(textVal);
        linkMask.push(linkVal);

        if (i < 20) {
          const idx = i / outC;
          console.log(`[#${idx}] raw: (${rawScore.toFixed(3)}, ${rawLink.toFixed(3)}) | sigm: (${score.toFixed(3)}, ${link.toFixed(3)})`);
        }
      }

      console.log(`🔥 Score range: ${scoreMin.toFixed(3)} - ${scoreMax.toFixed(3)}`);
      console.log(`🔥 Link range: ${linkMin.toFixed(3)} - ${linkMax.toFixed(3)}`);
      console.log("✅ textMask sample:", textMask.slice(0, 10));
      console.log("✅ linkMask sample:", linkMask.slice(0, 10));
      console.log("✅ textMask true count:", textTrueCount);
      console.log("✅ linkMask true count:", linkTrueCount);

      return {
        textMask,
        linkMask,
        outH,
        outW
      };
    };


    const runRecognizer = async (originalImageUri, textMask, outH, outW) => {
      console.log("🐞 runRecognizer() called");

      const boxes = extractBoundingBoxes(textMask, outW, outH);
      console.log("🐞 Boxes found:", boxes.length, boxes);

      if (boxes.length === 0) {
        console.warn("⚠️ No text regions found.");
        return [];
      }

      const recognizedTexts = [];
      const [, , recH, recW] = recognizer.model.inputs[0].shape;
      const [, timeSteps, numClasses] = recognizer.model.outputs[0].shape;

      for (const [i, box] of boxes.entries()) {
        console.log(`🐞 Cropping box #${i}:`, box);

        const cropped = await ImageResizer.createResizedImage(
          originalImageUri, recW, recH, 'JPEG', 100, 0,
          undefined, false, {
          mode: 'cover',
          onlyScaleDown: false,
          offset: { x: box.x, y: box.y },
          size: { width: box.width, height: box.height }
        }
        );

        const rgbArray = await rgb.convertToRGB(cropped.uri);
        const grayArray = rgbToGrayscale(rgbArray);
        const normalized = grayArray.map(v => (v - 127) / 255);
        const inputTensor = Float32Array.from(normalized);

        const recognizerOutput = await recognizer.model.run([inputTensor]);
        const output = recognizerOutput[0];

        const predictions = [];
        for (let t = 0; t < timeSteps; t++) {
          let maxProb = -Infinity, maxIndex = 0;
          for (let c = 0; c < numClasses; c++) {
            const val = output[t * numClasses + c];
            if (val > maxProb) {
              maxProb = val;
              maxIndex = c;
            }
          }
          predictions.push(maxIndex);
        }

        const deduped = predictions.filter((v, i, arr) => v !== 0 && (i === 0 || v !== arr[i - 1]));
        const decoded = deduped.map(i => alphabet[i - 1] || '').join('');
        console.log(`🐞 Box #${i} result:`, decoded);

        recognizedTexts.push(decoded);
      }

      console.log("🎉 Final recognized texts:", recognizedTexts);
      return recognizedTexts;
    };

    const pickImageFromGallery = async () => {
      console.log("📷 pickImageFromGallery called");
      try {
        const result = await launchImageLibrary({ mediaType: 'photo' });
        if (!result.assets || result.assets.length === 0) {
          console.warn("❌ No image selected");
          return;
        }

        const image = result.assets[0];
        console.log("📸 Picked image:", image.uri);

        const detectorResult = await runDetector(image.uri);
        const recognizedTexts = await runRecognizer(image.uri, detectorResult.textMask, detectorResult.outH, detectorResult.outW);

        console.log("✅ OCR Result:", recognizedTexts.join("\n"));
      } catch (error) {
        console.error("❌ Error during OCR:", error);
      }
    };

    function reshapeMask(mask, width, height) {
      const result = [];
      for (let y = 0; y < height; y++) {
        result[y] = [];
        for (let x = 0; x < width; x++) {
          result[y][x] = mask[y * width + x] ? 1 : 0;
        }
      }
      return result;
    }

    function floodFill(mask, visited, x, y, width, height, blob) {
      const queue = [[x, y]];
      while (queue.length) {
        const [cx, cy] = queue.pop();
        if (cx < 0 || cy < 0 || cx >= width || cy >= height) continue;
        if (visited[cy][cx] || !mask[cy][cx]) continue;
        visited[cy][cx] = true;
        blob.push([cx, cy]);
        queue.push([cx + 1, cy], [cx - 1, cy], [cx, cy + 1], [cx, cy - 1]);
      }
    }

    function extractBoundingBoxes(mask, width, height, minArea = 10) {
      console.log("🐞 extractBoundingBoxes called");
      const binaryMask = reshapeMask(mask, width, height);
      const visited = Array.from({ length: height }, () => Array(width).fill(false));
      const boxes = [];

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          if (binaryMask[y][x] && !visited[y][x]) {
            const blob = [];
            floodFill(binaryMask, visited, x, y, width, height, blob);
            if (blob.length < minArea) continue;
            const xs = blob.map(([x]) => x), ys = blob.map(([, y]) => y);
            boxes.push({
              x: Math.min(...xs),
              y: Math.min(...ys),
              width: Math.max(...xs) - Math.min(...xs) + 1,
              height: Math.max(...ys) - Math.min(...ys) + 1
            });
          }
        }
      }

      console.log("🐞 Total boxes extracted:", boxes.length);
      return boxes;
    }

    function rgbToGrayscale(rgbArray) {
      const gray = [];
      for (let i = 0; i < rgbArray.length; i += 3) {
        const r = rgbArray[i], g = rgbArray[i + 1], b = rgbArray[i + 2];
        gray.push(0.2989 * r + 0.5870 * g + 0.1140 * b);
      }
      return gray;
    }

    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <TouchableOpacity
          style={{ padding: 10, backgroundColor: 'black', borderRadius: 5, paddingHorizontal: 30 }}
          onPress={() => {
            console.log("🔁 Navigating to Camera screen");
            navigation.navigate('Camera');
          }}
        >
          <Text style={{ color: 'white', fontSize: 14 }}>Start</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={{
            padding: 15,
            backgroundColor: 'white',
            position: 'absolute',
            bottom: 110,
            borderRadius: 10,
            paddingHorizontal: 30
          }}
          onPress={pickImageFromGallery}
        >
          <Text>Pick from Gallery</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const Stack = createNativeStackNavigator();

  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="test">
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="Camera" component={CameraScreen} />
        <Stack.Screen name='main' component={MainScreen} />
        <Stack.Screen name='test' component={TestScreen}/>
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default App;
