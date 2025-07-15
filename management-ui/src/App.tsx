import { BrowserRouter, Routes, Route } from "react-router-dom";
import Apps from "./pages/Apps";
import Benchmarks from "./pages/Benchmarks";
import Layout from "./components/basic/Layout";

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Apps />} />
          <Route path="/apps" element={<Apps />} />
          <Route path="/benchmarks" element={<Benchmarks />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
