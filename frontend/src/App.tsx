import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Predict from "./pages/Predict";
import Explanation from "./pages/Explanation";
import Performance from "./pages/Performance";
import About from "./pages/About";

export default function App() {
  return (
    <>
      <Navbar />
      <main className="container">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/explanation" element={<Explanation />} />
          <Route path="/performance" element={<Performance />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
    </>
  );
}
