import React from 'react';
import { NavigationContainer, useNavigation } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { Text, TouchableOpacity, View } from "react-native";
import CameraScreen from "./CameraScreen";
import { launchImageLibrary } from 'react-native-image-picker';
import ImageResizer from '@bam.tech/react-native-image-resizer';
import * as rgb from 'react-native-image-to-rgb';
import { useTensorflowModel } from "react-native-fast-tflite";

const App = () => {
  function HomeScreen() {
    const navigation = useNavigation();

    const detector = useTensorflowModel(require('./assets/EasyOCR_EasyOCRDetector.tflite'));
    const recognizer = useTensorflowModel(require('./assets/EasyOCR_EasyOCRRecognizer.tflite'));
    const alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,- /_".split('');

    const pickImageFromGallery = async () => {
      try {
        const result = await launchImageLibrary({ mediaType: 'photo' });
        if (!result.assets || result.assets.length === 0) return;

        const image = result.assets[0];
        console.log("Picked image:", image.uri);

        // 📏 DETECTOR input shape
        const [, , detectorHeight, detectorWidth] = detector.model.inputs[0].shape;

        const resizedImage = await ImageResizer.createResizedImage(
          image.uri, detectorWidth, detectorHeight, 'JPEG', 100, 0
        );
        const imageRgb = await rgb.convertToRGB(resizedImage.uri); // flat RGB array

        const chw = [[], [], []];
        for (let i = 0; i < imageRgb.length; i += 3) {
          chw[0].push((imageRgb[i] - 127) / 255);     // R
          chw[1].push((imageRgb[i + 1] - 127) / 255); // G
          chw[2].push((imageRgb[i + 2] - 127) / 255); // B
        }

        const detectorInput = Float32Array.from(chw.flat());
        const detectorOutput = await detector.model.run([detectorInput]);

        const [, outputHeight, outputWidth, outputChannels] = detector.model.outputs[0].shape;
        console.log("Detector output shape:", [outputHeight, outputWidth, outputChannels]);

        // Optional: Simulated detection points
        const outputTensor = detectorOutput[0];
        const topScores = [];
        for (let y = 0; y < outputHeight; y++) {
          for (let x = 0; x < outputWidth; x++) {
            const idx = (y * outputWidth + x) * outputChannels;
            const score = outputTensor[idx]; // assuming score is in channel 0
            if (score > 0.5) {
              topScores.push({ x, y, score });
            }
          }
        }

        topScores.sort((a, b) => b.score - a.score);
        const top = topScores.slice(0, 5);
        console.log("Top detection points (x, y, score):", top);

        // 👁️ RECOGNIZER input shape
        const [, , recognizerHeight, recognizerWidth] = recognizer.model.inputs[0].shape;

        const fakeCropped = await ImageResizer.createResizedImage(
          image.uri, recognizerWidth, recognizerHeight, 'JPEG', 100, 0
        );

        const rgbGray = await rgb.convertToRGB(fakeCropped.uri);
        const gray = rgbToGrayscale(rgbGray);
        const grayNormalized = gray.map(v => (v - 127) / 255);
        const recognizerInput = Float32Array.from(grayNormalized);

        const recognizerOutput = await recognizer.model.run([recognizerInput]);
        const output = recognizerOutput[0]; // shape: [1, T, C] → flat array

        const [, timeSteps, numClasses] = recognizer.model.outputs[0].shape;

        const predictions = [];
        for (let t = 0; t < timeSteps; t++) {
          let maxProb = -Infinity;
          let maxIndex = 0;
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
        const recognizedText = deduped.map(i => alphabet[i - 1] || '').join('');
        console.log("✅ Recognized Text:", recognizedText);

      } catch (error) {
        console.error("OCR error:", error);
      }
    };

    function rgbToGrayscale(rgbArray) {
      const gray = [];
      for (let i = 0; i < rgbArray.length; i += 3) {
        const r = rgbArray[i];
        const g = rgbArray[i + 1];
        const b = rgbArray[i + 2];
        const grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b;
        gray.push(grayscale);
      }
      return gray;
    }

    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <TouchableOpacity
          style={{ padding: 10, backgroundColor: 'black', borderRadius: 5, paddingHorizontal: 30 }}
          onPress={() => navigation.navigate('Camera')}
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
      <Stack.Navigator initialRouteName="Home">
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="Camera" component={CameraScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default App;
