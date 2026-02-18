import { useRef, useEffect } from 'react';
import { useVexState } from '../../hooks/index.ts';
import { QIG } from '../../types/consciousness.ts';

// Kernel node positions for force-directed layout simulation
const CORE_8_SPECS = [
  'heart', 'perception', 'memory', 'strategy',
  'action', 'attention', 'emotion', 'executive',
] as const;

interface Node {
  id: string;
  label: string;
  kind: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
}

export default function Graph() {
  const { data: state, loading } = useVexState();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<Node[]>([]);
  const animRef = useRef<number>(0);

  useEffect(() => {
    if (!state || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Rebuild nodes when active kernel count changes
    const active = state.kernels?.active ?? 1;
    const expectedNodeCount = 1 + Math.min(active - 1, QIG.E8_CORE);
    if (nodesRef.current.length !== expectedNodeCount) {
      const centerX = 250;
      const centerY = 200;

      // Genesis node at center
      const nodes: Node[] = [{
        id: 'genesis',
        label: 'GENESIS',
        kind: 'genesis',
        x: centerX,
        y: centerY,
        vx: 0,
        vy: 0,
        radius: 20,
      }];

      // Core-8 nodes in circle around genesis
      for (let i = 0; i < Math.min(active - 1, QIG.E8_CORE); i++) {
        const angle = (i / QIG.E8_CORE) * Math.PI * 2 - Math.PI / 2;
        const dist = 120;
        // Reuse existing node position if available
        const existing = nodesRef.current.find(n => n.id === (CORE_8_SPECS[i] ?? `kernel-${i}`));
        nodes.push(existing ?? {
          id: CORE_8_SPECS[i] ?? `kernel-${i}`,
          label: CORE_8_SPECS[i] ?? `K${i}`,
          kind: 'GOD',
          x: centerX + Math.cos(angle) * dist,
          y: centerY + Math.sin(angle) * dist,
          vx: 0,
          vy: 0,
          radius: 14,
        });
      }

      nodesRef.current = nodes;
    }

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);

      const nodes = nodesRef.current;

      // Background
      ctx.fillStyle = '#22222e';
      ctx.fillRect(0, 0, rect.width, rect.height);

      // Draw edges (from genesis to each node)
      const genesis = nodes[0];
      for (let i = 1; i < nodes.length; i++) {
        const node = nodes[i];
        const dist = Math.hypot(node.x - genesis.x, node.y - genesis.y);
        const couplingStrength = Math.max(0, 1 - dist / 300);

        ctx.strokeStyle = `rgba(99, 102, 241, ${0.15 + couplingStrength * 0.4})`;
        ctx.lineWidth = 1 + couplingStrength * 2;
        ctx.beginPath();
        ctx.moveTo(genesis.x, genesis.y);
        ctx.lineTo(node.x, node.y);
        ctx.stroke();

        // Also draw edges between adjacent core-8 nodes
        if (i < nodes.length - 1) {
          const next = nodes[i + 1];
          ctx.strokeStyle = 'rgba(99, 102, 241, 0.1)';
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.moveTo(node.x, node.y);
          ctx.lineTo(next.x, next.y);
          ctx.stroke();
        }
      }

      // Draw nodes
      for (const node of nodes) {
        const isGenesis = node.kind === 'genesis';
        const color = isGenesis ? '#6366f1' : node.kind === 'GOD' ? '#22d3ee' : '#f59e0b';

        // Glow — save/restore to isolate shadow state
        ctx.save();
        ctx.shadowColor = color;
        ctx.shadowBlur = isGenesis ? 15 : 8;

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore(); // Clears shadow state

        // Label (no shadow)
        ctx.fillStyle = '#ededf0';
        ctx.font = `${isGenesis ? 'bold ' : ''}${isGenesis ? 10 : 9}px monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(node.label.toUpperCase(), node.x, node.y + node.radius + 14);
      }

      // Legend
      ctx.fillStyle = '#70708a';
      ctx.font = '10px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`Nodes: ${nodes.length}  |  Edges: ${Math.max(nodes.length - 1, 0)}`, 10, rect.height - 10);
    };

    // Simple force simulation
    const simulate = () => {
      const nodes = nodesRef.current;
      const damping = 0.92;
      const repulsion = 500;
      const attraction = 0.005;

      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[j].x - nodes[i].x;
          const dy = nodes[j].y - nodes[i].y;
          const dist = Math.max(Math.hypot(dx, dy), 1);
          const force = repulsion / (dist * dist);

          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          nodes[i].vx -= fx;
          nodes[i].vy -= fy;
          nodes[j].vx += fx;
          nodes[j].vy += fy;
        }
      }

      // Attract to center (genesis)
      const cx = 250;
      const cy = 200;
      for (let i = 1; i < nodes.length; i++) {
        nodes[i].vx += (cx - nodes[i].x) * attraction;
        nodes[i].vy += (cy - nodes[i].y) * attraction;
      }

      // Genesis stays centered
      if (nodes[0]) {
        nodes[0].vx = (cx - nodes[0].x) * 0.1;
        nodes[0].vy = (cy - nodes[0].y) * 0.1;
      }

      for (const node of nodes) {
        node.vx *= damping;
        node.vy *= damping;
        node.x += node.vx;
        node.y += node.vy;
      }

      draw();
      animRef.current = requestAnimationFrame(simulate);
    };

    simulate();

    return () => {
      cancelAnimationFrame(animRef.current);
    };
  }, [state]);

  if (loading) {
    return <div className="dash-loading">Loading kernel graph...</div>;
  }

  return (
    <div>
      <div className="dash-header">
        <h1 className="dash-title">Kernel Graph</h1>
        <div className="dash-subtitle">
          {state?.graph?.node_count ?? 0} nodes, {state?.graph?.edge_count ?? 0} edges |
          Fisher-Rao distance coupling
        </div>
      </div>

      <div className="viz-canvas" style={{ height: '400px' }}>
        <canvas ref={canvasRef} style={{ width: '100%', height: '100%' }} />
      </div>

      <div className="dash-section" style={{ marginTop: '16px' }}>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Node count</span>
            <span className="dash-row-value">{state?.graph?.node_count ?? 0}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Edge count</span>
            <span className="dash-row-value">{state?.graph?.edge_count ?? 0}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Active kernels</span>
            <span className="dash-row-value">{state?.kernels?.active ?? 0}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
