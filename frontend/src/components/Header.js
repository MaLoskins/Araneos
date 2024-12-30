import React from 'react';
import { NavLink } from 'react-router-dom';
import logo from '../assets/logo.svg';

const Header = () => {
  return (
    <header className="app-header">
      <div className="logo-container">
        <img src={logo} alt="Logo" className="app-logo" />
        <span className="app-name">Graphcentric</span>
      </div>
      <nav className="nav-links">
        <NavLink
          to="/"
          end
          className={({ isActive }) =>
            isActive ? 'nav-link active-link' : 'nav-link'
          }
        >
          GraphNet
        </NavLink>
      </nav>
    </header>
  );
};

export default Header;
