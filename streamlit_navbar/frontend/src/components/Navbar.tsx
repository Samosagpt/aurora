import React from 'react';
import { NavItem } from '../types.ts';

interface NavbarProps {
  items: NavItem[];
  logoText: string;
  logoImage?: string;
  selected?: string;
  sticky: boolean;
  style: React.CSSProperties;
  onItemClick: (itemId: string) => void;
}

const Navbar: React.FC<NavbarProps> = ({
  items,
  logoText,
  logoImage,
  selected,
  sticky,
  style,
  onItemClick,
}) => {
  return (
    <nav 
      className={`navbar ${sticky ? 'sticky' : ''}`}
      style={style}
    >
      <div className="navbar-logo">
        {logoImage ? (
          <img src={logoImage} alt="Logo" className="navbar-logo-image" />
        ) : (
          logoText
        )}
      </div>
      
      <ul className="navbar-nav">
        {items.map((item) => (
          <li
            key={item.id}
            className={`nav-item ${selected === item.id ? 'active' : ''} ${
              item.disabled ? 'disabled' : ''
            }`}
            onClick={() => !item.disabled && onItemClick(item.id)}
          >
            {item.icon && (
              <span className="nav-item-icon">{item.icon}</span>
            )}
            <span className="nav-item-label">{item.label}</span>
          </li>
        ))}
      </ul>
    </nav>
  );
};

export default Navbar;
