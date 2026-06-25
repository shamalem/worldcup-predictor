import {
  BarChart, Bar, XAxis, YAxis, Cell, ResponsiveContainer, Tooltip, ReferenceLine,
} from "recharts";
import type { FeatureContribution } from "../types";

export default function ContributionChart({ data }: { data: FeatureContribution[] }) {
  const top = [...data]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 8)
    .reverse();

  return (
    <ResponsiveContainer width="100%" height={Math.max(260, top.length * 38)}>
      <BarChart data={top} layout="vertical" margin={{ left: 8, right: 24 }}>
        <XAxis type="number" stroke="#8aa597" tick={{ fontSize: 11 }} />
        <YAxis type="category" dataKey="label" width={210}
          stroke="#8aa597" tick={{ fontSize: 11 }} />
        <ReferenceLine x={0} stroke="#21362c" />
        <Tooltip
          contentStyle={{ background: "#16271f", border: "1px solid #21362c", borderRadius: 8, color: "#eef3ef" }}
          formatter={(v: number) => [v.toFixed(3), "contribution"]} />
        <Bar dataKey="contribution" radius={4}>
          {top.map((d, i) => (
            <Cell key={i} fill={d.contribution >= 0 ? "#2ee27a" : "#e2725b"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
