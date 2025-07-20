import React from 'react';
import ReactDOM from 'react-dom/client';
import StreamlitNavbar from './StreamlitNavbar.tsx';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

root.render(
  <React.StrictMode>
    <StreamlitNavbar />
  </React.StrictMode>
);
