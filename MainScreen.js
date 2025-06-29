import React from 'react';
import { NavigationContainer, useNavigation } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { Text, TouchableOpacity, View } from "react-native";
import CameraScreen from "./CameraScreen";
import { launchImageLibrary } from 'react-native-image-picker';
import ImageResizer from '@bam.tech/react-native-image-resizer';
import * as rgb from 'react-native-image-to-rgb';
import { useTensorflowModel } from "react-native-fast-tflite";

const MainScreen = () => {
    const navigation = useNavigation();

    const detector = useTensorflowModel(require('./assets/EasyOCR_EasyOCRDetector.tflite'));
    const recognizer = useTensorflowModel(require('./assets/EasyOCR_EasyOCRRecognizer.tflite'));

    const preprocessDetectorImage = async (imageUri, inputShape, inputDtype, inputQuant) => {
        const [batch, d1, d2, d3] = inputShape;

        // Determine layout
        let layout, targetHeight, targetWidth, channels;
        if ([1, 3].includes(d3)) {
            layout = 'NHWC';
            targetHeight = d1;
            targetWidth = d2;
            channels = d3;
        } else if ([1, 3].includes(d1)) {
            layout = 'NCHW';
            targetHeight = d2;
            targetWidth = d3;
            channels = d1;
        } else {
            throw new Error(`Cannot infer layout from inputShape: ${inputShape}`);
        }

        // ✅ Resize image with mode: 'stretch' to ensure exact dimensions
        const resizedImage = await ImageResizer.createResizedImage(
            imageUri,
            targetWidth,
            targetHeight,
            'JPEG',
            100,
            0,        // rotation
            undefined, // outputPath
            false,     // keepExif
            { mode: 'stretch' }  // ✅ Enforce exact size
        );

        const { width: resizedWidth, height: resizedHeight } = resizedImage;

        const rgbData = await rgb.convertToRGB(resizedImage.uri);

        if (!rgbData || rgbData.length !== resizedWidth * resizedHeight * 3) {
            throw new Error(`Invalid RGB data size. Expected ${resizedWidth * resizedHeight * 3}, got ${rgbData.length}`);
        }

        if (resizedWidth !== targetWidth || resizedHeight !== targetHeight) {
            throw new Error(`Image was resized to ${resizedWidth}x${resizedHeight}, expected ${targetWidth}x${targetHeight}`);
        }

        const totalElements = channels * targetHeight * targetWidth;
        const isFloat = inputDtype === 'float32' || inputDtype === 'float';
        const outputArray = isFloat
            ? new Float32Array(totalElements)
            : new Uint8Array(totalElements);  // for quantized input

        const [scale, zeroPoint] = inputQuant || [0, 0];

        for (let i = 0; i < targetHeight; i++) {
            for (let j = 0; j < targetWidth; j++) {
                const pixelIdx = (i * targetWidth + j) * 3;
                const r = rgbData[pixelIdx] / 255.0;
                const g = rgbData[pixelIdx + 1] / 255.0;
                const b = rgbData[pixelIdx + 2] / 255.0;

                const baseIdx = i * targetWidth + j;

                if (layout === 'NCHW') {
                    const idxR = 0 * targetHeight * targetWidth + baseIdx;
                    const idxG = 1 * targetHeight * targetWidth + baseIdx;
                    const idxB = 2 * targetHeight * targetWidth + baseIdx;

                    if (isFloat) {
                        outputArray[idxR] = r;
                        outputArray[idxG] = g;
                        outputArray[idxB] = b;
                    } else {
                        outputArray[idxR] = Math.round(r / scale + zeroPoint);
                        outputArray[idxG] = Math.round(g / scale + zeroPoint);
                        outputArray[idxB] = Math.round(b / scale + zeroPoint);
                    }
                } else {
                    const idx = (i * targetWidth + j) * 3;
                    if (isFloat) {
                        outputArray[idx] = r;
                        outputArray[idx + 1] = g;
                        outputArray[idx + 2] = b;
                    } else {
                        outputArray[idx] = Math.round(r / scale + zeroPoint);
                        outputArray[idx + 1] = Math.round(g / scale + zeroPoint);
                        outputArray[idx + 2] = Math.round(b / scale + zeroPoint);
                    }
                }
            }
        }

        return {
            inputTensor: {
                shape: inputShape,
                data: outputArray,
                type: inputDtype
            },
            resizedSize: {
                width: targetWidth,
                height: targetHeight
            }
        };
    };




    const runDetector = async (imageUri) => {
        console.log("🔍 Starting detector...");

        // Get detector input shape
        const detectorInputShape = detector.model.inputs[0].shape;
        // console.log("🖼️ Detector input shape:", detectorInputShape);

        let det_shape = detector.model.inputs[0].shape;
        let det_dtype = detector.model.inputs[0].dataType;
        let det_quant = [0.0, 0];

        console.log('det_shape', det_shape);
        console.log('det_dtype', det_dtype);
        console.log('det_quant', det_quant);

        const { inputTensor, resizedSize } = await preprocessDetectorImage(
            imageUri,
            det_shape,
            det_dtype,
        );
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
            console.log("📸 Picked image:", image.uri, "Size:", image.width, "x", image.height);

            // Run detector
            const detectorResult = await runDetector(image.uri);

            // // Post-process detector output
            // const combinedMask = postprocessScoreLink(
            //   detectorResult.outputMaps,
            //   detectorResult.outH,
            //   detectorResult.outW,
            //   detectorResult.outC,
            //   0.7, // text_threshold
            //   0.4  // link_threshold
            // );

            // // Find connected components
            // const boxes = findConnectedBoxes(
            //   combinedMask,
            //   detectorResult.outW,
            //   detectorResult.outH,
            //   10 // min_area
            // );

            // // Map boxes to original image coordinates
            // const originalSize = { width: image.width, height: image.height };
            // const maskSize = { width: detectorResult.outW, height: detectorResult.outH };

            // const mappedBoxes = boxes.map(box =>
            //   mapBoxToOriginal(box, maskSize, detectorResult.resizedSize, originalSize)
            // );

            // console.log("📦 Mapped boxes:", mappedBoxes);

            // // Run recognizer on each box
            // const recognizedTexts = await runRecognizer(image.uri, mappedBoxes, originalSize);

            // console.log("✅ Final OCR Result:", recognizedTexts.join("\n"));

            // // You can update UI state here to show results

        } catch (error) {
            console.error("❌ Error during OCR:", error);
        }
    };

    return (
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
            <TouchableOpacity
                style={{ padding: 10, backgroundColor: 'black', borderRadius: 5, paddingHorizontal: 30 }}
                onPress={() => {
                    console.log("🔁 Navigating to Camera screen");
                    navigation.navigate('Camera');
                }}
            >
                <Text style={{ color: 'white', fontSize: 14 }}>Start Camera</Text>
            </TouchableOpacity>
            <TouchableOpacity
                style={{
                    padding: 15,
                    backgroundColor: 'blue',
                    position: 'absolute',
                    bottom: 110,
                    borderRadius: 10,
                    paddingHorizontal: 30,
                    marginTop: 20
                }}
                onPress={pickImageFromGallery}
            >
                <Text style={{ color: 'white' }}>Pick from Gallery</Text>
            </TouchableOpacity>
        </View>
    );
}

export default MainScreen;