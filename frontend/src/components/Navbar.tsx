import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Home", end: true },
  { to: "/predict", label: "Predict" },
  { to: "/explanation", label: "Explanation" },
  { to: "/performance", label: "Model" },
  { to: "/about", label: "About" },
];

export default function Navbar() {
  return (
    <nav className="nav">
      <div className="brand">
        <span className="pitch-dot" />
        <span>WC&nbsp;Predictor</span>
      </div>
      <div className="nav-links">
        {links.map((l) => (
          <NavLink key={l.to} to={l.to} end={l.end}
            className={({ isActive }) => (isActive ? "active" : "")}>
            {l.label}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
