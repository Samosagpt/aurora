import React, { useEffect, useState } from 'react';
import {
  Streamlit,
  withStreamlitConnection,
  ComponentProps,
} from 'streamlit-component-lib';
import Navbar from './components/Navbar.tsx';
import { NavItem } from './types.ts';
import './index.css';

interface StreamlitArgs {
  items: NavItem[];
  logoText: string;
  logoImage?: string;
  selected?: string;
  sticky: boolean;
  style: React.CSSProperties;
}

const StreamlitNavbar: React.FC<ComponentProps> = ({ args }) => {
  const {
    items = [],
    logoText = "🌅 Aurora",
    logoImage,
    selected,
    sticky = true,
    style = {},
  }: StreamlitArgs = args;

  const [selectedItem, setSelectedItem] = useState<string>(selected || "");
  const [isDarkMode, setIsDarkMode] = useState<boolean>(true); // Default to dark mode

  useEffect(() => {
    Streamlit.setFrameHeight(80);
    Streamlit.setComponentReady();
  }, []);

  useEffect(() => {
    // Detect theme changes
    const detectTheme = () => {
      const streamlitDoc = window.parent?.document;
      if (streamlitDoc) {
        const isDark = streamlitDoc.documentElement.getAttribute('data-theme') === 'dark' ||
                      streamlitDoc.body.classList.contains('css-1d391kg') ||
                      window.matchMedia('(prefers-color-scheme: dark)').matches;
        setIsDarkMode(isDark);
      }
    };

    detectTheme();
    
    // Listen for theme changes
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    mediaQuery.addEventListener('change', detectTheme);
    
    // Check for theme changes periodically (for Streamlit theme switching)
    const interval = setInterval(detectTheme, 1000);

    return () => {
      mediaQuery.removeEventListener('change', detectTheme);
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    if (selected !== undefined) {
      setSelectedItem(selected);
    }
  }, [selected]);

  const handleItemClick = (itemId: string) => {
    setSelectedItem(itemId);
    Streamlit.setComponentValue(itemId);
  };

  return (
    <div 
      className={`navbar-container ${isDarkMode ? 'dark-mode' : 'light-mode'}`}
      style={{ 
        width: '100%', 
        height: '80px',
        margin: 0, 
        padding: 0,
        display: 'flex',
        alignItems: 'center'
      }}
    >
      <Navbar
        items={items}
        logoText={logoText}
        logoImage={logoImage}
        selected={selectedItem}
        sticky={sticky}
        style={style}
        onItemClick={handleItemClick}
      />
    </div>
  );
};

export default withStreamlitConnection(StreamlitNavbar);
