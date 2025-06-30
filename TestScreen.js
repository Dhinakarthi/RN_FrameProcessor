import React from 'react';
import { NavigationContainer, useNavigation } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { Text, TouchableOpacity, View } from "react-native";
import CameraScreen from "./CameraScreen";
import { launchImageLibrary } from 'react-native-image-picker';
import ImageResizer from '@bam.tech/react-native-image-resizer';
import * as rgb from 'react-native-image-to-rgb';
import { useTensorflowModel } from "react-native-fast-tflite";

const TestScreen = () => {
    const navigation = useNavigation();

    const detector = useTensorflowModel(require('./assets/EasyOCR_EasyOCRDetector.tflite'));
    const recognizer = useTensorflowModel(require('./assets/EasyOCR_EasyOCRRecognizer.tflite'));

    // const preprocessDetectorImage = async (imageUri, det_shape, det_dtype, det_quant) => {
    //     const [batch, d1, d2, d3] = det_shape;

    //     // Determine layout
    //     let layout, targetHeight, targetWidth, channels;
    //     if ([1, 3].includes(d3)) {
    //         layout = 'NHWC';
    //         targetHeight = d1;
    //         targetWidth = d2;
    //         channels = d3;
    //     } else if ([1, 3].includes(d1)) {
    //         layout = 'NCHW';
    //         targetHeight = d2;
    //         targetWidth = d3;
    //         channels = d1;
    //     } else {
    //         throw new Error(`Cannot infer layout from det_shape: ${det_shape}`);
    //     }

    //     // ✅ Resize image with mode: 'stretch' to ensure exact dimensions
    //     const resizedImage = await ImageResizer.createResizedImage(
    //         imageUri,
    //         targetWidth,
    //         targetHeight,
    //         'JPEG',
    //         100,
    //         0,        // rotation
    //         undefined, // outputPath
    //         false,     // keepExif
    //         { mode: 'stretch' }  // ✅ Enforce exact size
    //     );

    //     const { width: resizedWidth, height: resizedHeight } = resizedImage;

    //     const rgbData = await rgb.convertToRGB(resizedImage.uri);

    //     if (!rgbData || rgbData.length !== resizedWidth * resizedHeight * 3) {
    //         throw new Error(`Invalid RGB data size. Expected ${resizedWidth * resizedHeight * 3}, got ${rgbData.length}`);
    //     }

    //     if (resizedWidth !== targetWidth || resizedHeight !== targetHeight) {
    //         throw new Error(`Image was resized to ${resizedWidth}x${resizedHeight}, expected ${targetWidth}x${targetHeight}`);
    //     }

    //     const totalElements = channels * targetHeight * targetWidth;
    //     const isFloat = det_dtype === 'float32' || det_dtype === 'float';
    //     const outputArray = isFloat
    //         ? new Float32Array(totalElements)
    //         : new Uint8Array(totalElements);  // for quantized input

    //     const [scale, zeroPoint] = det_quant || [0, 0];

    //     for (let i = 0; i < targetHeight; i++) {
    //         for (let j = 0; j < targetWidth; j++) {
    //             const pixelIdx = (i * targetWidth + j) * 3;
    //             const r = rgbData[pixelIdx] / 255.0;
    //             const g = rgbData[pixelIdx + 1] / 255.0;
    //             const b = rgbData[pixelIdx + 2] / 255.0;

    //             const baseIdx = i * targetWidth + j;

    //             if (layout === 'NCHW') {
    //                 const idxR = 0 * targetHeight * targetWidth + baseIdx;
    //                 const idxG = 1 * targetHeight * targetWidth + baseIdx;
    //                 const idxB = 2 * targetHeight * targetWidth + baseIdx;

    //                 if (isFloat) {
    //                     outputArray[idxR] = r;
    //                     outputArray[idxG] = g;
    //                     outputArray[idxB] = b;
    //                 } else {
    //                     outputArray[idxR] = Math.round(r / scale + zeroPoint);
    //                     outputArray[idxG] = Math.round(g / scale + zeroPoint);
    //                     outputArray[idxB] = Math.round(b / scale + zeroPoint);
    //                 }
    //             } else {
    //                 const idx = (i * targetWidth + j) * 3;
    //                 if (isFloat) {
    //                     outputArray[idx] = r;
    //                     outputArray[idx + 1] = g;
    //                     outputArray[idx + 2] = b;
    //                 } else {
    //                     outputArray[idx] = Math.round(r / scale + zeroPoint);
    //                     outputArray[idx + 1] = Math.round(g / scale + zeroPoint);
    //                     outputArray[idx + 2] = Math.round(b / scale + zeroPoint);
    //                 }
    //             }
    //         }
    //     }

    //     return {
    //         inputTensor: {
    //             shape: det_shape,
    //             data: outputArray,
    //             type: det_dtype
    //         },
    //         resizedSize: {
    //             width: targetWidth,
    //             height: targetHeight
    //         }
    //     };
    // };


    const preprocessDetectorImage = async (imageUri, det_shape, det_dtype, det_quant) => {
        if (det_shape.length !== 4) {
            throw new Error(`Detector input shape not 4D: ${det_shape}`);
        }

        const [batch, d1, d2, d3] = det_shape;
        let layout, targetH, targetW, channels;

        // Determine layout: NHWC or NCHW
        if ([1, 3].includes(d3)) {
            layout = 'NHWC';
            targetH = d1;
            targetW = d2;
            channels = d3;
        } else if ([1, 3].includes(d1)) {
            layout = 'NCHW';
            targetH = d2;
            targetW = d3;
            channels = d1;
        } else {
            throw new Error(`Cannot infer detector layout from shape ${det_shape}`);
        }

        // Step 1: Resize image
        const resized = await ImageResizer.createResizedImage(
            imageUri,
            targetW,
            targetH,
            'JPEG',
            100
        );

        // Step 2: Convert to RGB pixel array
        // Result is a Uint8Array: [R,G,B, R,G,B, ..., R,G,B] of size H*W*3
        const rgbData = await rgb.convertToRGB(resized.uri); // returns Uint8Array

        const numPixels = targetH * targetW;
        const floatData = new Float32Array(numPixels * channels);

        if (channels === 1) {
            // Convert RGB to grayscale (single channel)
            for (let i = 0; i < numPixels; i++) {
                const r = rgbData[i * 3] / 255;
                const g = rgbData[i * 3 + 1] / 255;
                const b = rgbData[i * 3 + 2] / 255;
                floatData[i] = 0.299 * r + 0.587 * g + 0.114 * b;
            }
        } else {
            // Normalize RGB
            for (let i = 0; i < numPixels * 3; i++) {
                floatData[i] = rgbData[i] / 255;
            }
        }

        // Step 3: Layout conversion
        let processed;
        if (layout === 'NHWC') {
            processed = floatData; // already (H, W, C)
        } else {
            // Convert (H, W, C) → (C, H, W)
            processed = new Float32Array(floatData.length);
            for (let c = 0; c < channels; c++) {
                for (let h = 0; h < targetH; h++) {
                    for (let w = 0; w < targetW; w++) {
                        const dst = c * targetH * targetW + h * targetW + w;
                        const src = h * targetW * channels + w * channels + c;
                        processed[dst] = floatData[src];
                    }
                }
            }
        }

        // Step 4: Quantize or cast
        const [scale, zeroPoint] = det_quant;
        let finalData;

        if (det_dtype === 'float32') {
            finalData = processed;
        } else {
            const quantized = new Uint8Array(processed.length);
            for (let i = 0; i < processed.length; i++) {
                let q = Math.round(processed[i] / scale + zeroPoint);
                quantized[i] = Math.max(0, Math.min(255, q));
            }
            finalData = quantized;
        }

        return [finalData, [targetW, targetH]];
    }

    const runModal = async (inputTensor) => {
        // Fix: model expects inputTensor in an array
        const result = await detector.model.run([inputTensor]);
        return result;
    };

    const runDetector = async (imageUri) => {
        try {
            console.log("🔍 Starting detector...");

            const det_shape = detector.model.inputs[0].shape;
            const det_dtype = detector.model.inputs[0].dataType;
            const det_quant = [0.0, 0];

            console.log("🖼️ Detector input shape:", det_shape);
            console.log("📦 Detector dtype:", det_dtype);
            console.log("📏 Detector quant:", det_quant);

            const [tensor, [w, h]] = await preprocessDetectorImage(
                imageUri,
                det_shape,
                det_dtype,
                det_quant
            );

            console.log("✅ Preprocessed image size:", w, h);
            console.log("📏 Tensor length:", tensor.length);
            console.log("🔍 First 10 values:", Array.from(tensor.slice(0, 10)));

            const output = await runModal(tensor); // ✅ tensor wrapped inside runModal

            // 🔒 Safely log output preview
            if (Array.isArray(output)) {
                console.log("✅ Output has", output.length, "tensors");

                output.forEach((tensorOut, idx) => {
                    const values = Object.values(tensorOut).slice(0, 10);
                    console.log(`🔍 Output[${idx}] first 10 values:`, values);
                });

            } else if (output instanceof Object && output.data) {
                const preview = Array.from(output.data.slice(0, 10));
                console.log("✅ Output shape:", output.data.length);
                console.log("🔍 First 10 output values:", preview);

            } else {
                console.log("⚠️ Output (raw):", output);
            }

        } catch (error) {
            console.error("❌ Error during detector run:", error);
        }
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

export default TestScreen;