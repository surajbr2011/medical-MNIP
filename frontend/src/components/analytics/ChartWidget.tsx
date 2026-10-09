import React, { type ReactNode } from 'react';

interface ChartWidgetProps {
  title: string;
  children: ReactNode;
  heightClass?: string;
  footer?: ReactNode;
}

export default function ChartWidget({ title, children, heightClass = 'h-72', footer }: ChartWidgetProps) {
  return (
    <div className="bg-[#161b22] border border-[#30363d] p-6 rounded-xl flex flex-col">
      <h3 className="text-lg font-semibold mb-6">{title}</h3>
      <div className={`w-full ${heightClass}`}>
        {children}
      </div>
      {footer && <div className="mt-4">{footer}</div>}
    </div>
  );
}
