import { useEffect, useRef } from "react";
import { Image, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { Camera, useCameraDevice, useCameraPermission } from "react-native-vision-camera";
import ImageResizer from '@bam.tech/react-native-image-resizer';
import * as rgb from 'react-native-image-to-rgb';
import { useTensorflowModel } from "react-native-fast-tflite";


const CameraScreen = () => {
    const { hasPermission, requestPermission } = useCameraPermission();
    const device = useCameraDevice('back');
    const modal = useTensorflowModel(require('./assets/EasyOCR_EasyOCRRecognizer.tflite'));

    const cameraRef = useRef();

    useEffect(() => {
        if (!hasPermission) {
            requestPermission()
        }
    }, [hasPermission]);

    const takePhoto = async () => {
        try {
            const photo = await cameraRef.current.takePhoto();

            console.log('Photo ===>', photo.path);

            const resizedImage = await ImageResizer.createResizedImage(photo?.path, 640, 640, 'JPEG', 100, 0);

            console.log('resizedImage', resizedImage?.uri)

            const imageRgb = await rgb.convertToRGB(resizedImage?.uri);

            // console.log('imageRgb length:', imageRgb.length);
            // console.log('imageRgb first 10 values:', imageRgb.slice(0, 10));

            const input = Float32Array.from(imageRgb);

            console.log('input length:', input.length);
            console.log('input first 10 values:', input.slice(0, 10));

            const output = await modal.model.run([input]);

            console.log('Output length:', output.length);

            if (output.length > 0) {
                const firstOutput = output[0];
                const keys = Object.keys(firstOutput);

                console.log('Output keys count:', keys.length);

                const outputArray = keys
                    .sort((a, b) => Number(a) - Number(b))
                    .map(key => firstOutput[key]);

                console.log('First 10 output values:', outputArray.slice(0, 10));
            }

        } catch (error) {
            console.log('Camera Error ===>', error);
        }
    }

    if (!hasPermission) return null;

    if (device == null) {
        return null;
    }

    return (
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
            <Camera
                ref={cameraRef}
                style={StyleSheet.absoluteFill}
                device={device}
                isActive={true}
                photo={true}
            />
            <TouchableOpacity
                style={{ padding: 15, backgroundColor: 'white', position: 'absolute', bottom: 50, borderRadius: 10, paddingHorizontal: 30 }}
                onPress={takePhoto}
            >
                <Text>Capture</Text>
            </TouchableOpacity>
        </View>
    )
}

export default CameraScreen;