import { NavigationContainer, useNavigation } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { Text, TouchableOpacity, View } from "react-native";
import CameraScreen from "./CameraScreen";

const App = () => {



  function HomeScreen() {
    const navigation = useNavigation();
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <TouchableOpacity
          style={{ padding: 10, backgroundColor: 'black', borderRadius: 5, paddingHorizontal: 30 }}
          onPress={() => navigation.navigate('Camera')}
        >
          <Text style={{ color: 'white', fontSize: 14 }}>Start</Text>
        </TouchableOpacity>
      </View>
    )
  }

  const Stack = createNativeStackNavigator();

  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Home">
        <Stack.Screen name="Home" component={HomeScreen}/>
        <Stack.Screen name="Camera" component={CameraScreen}/>
      </Stack.Navigator>
    </NavigationContainer>
  )
}

export default App;