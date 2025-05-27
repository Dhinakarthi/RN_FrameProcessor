import { Text, TouchableOpacity, View } from "react-native";

const App = () => {
  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <TouchableOpacity style={{ padding: 10, backgroundColor: 'black', borderRadius: 5, paddingHorizontal: 30 }}>
        <Text style={{ color: 'white', fontSize: 14 }}>Start</Text>
      </TouchableOpacity>
    </View>
  )
}

export default App;