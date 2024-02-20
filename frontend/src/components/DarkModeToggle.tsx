
import React from 'react';

const DarkModeToggle = () => {
  const [darkMode, setDarkMode] = React.useState(() => {
    // Check for dark mode preference at the OS level
    const osPrefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    // Check user's preference from localStorage if available
    const userPrefersDark = localStorage.getItem('theme') ? localStorage.getItem('theme') === 'dark' : osPrefersDark;
    return userPrefersDark;
  });

  React.useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [darkMode]);

  return (
    <button onClick={() => setDarkMode(!darkMode)}>
      {darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
    </button>
  );
};

export default DarkModeToggle;