// /** @type {import('tailwindcss').Config} */
// export default {
//   content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
//   theme: {
//     extend: {
//       colors: {
//         customNavy: 'rgb(0, 0, 40)',
//         customBlue: 'rgb(0, 152, 255)', // Adding custom text/icon color
//       },
//     },
//     container: {
//       padding: "7rem",
//       center: true,
//     },
//   },
//   plugins: [require("@tailwindcss/typography")],
// };
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: 'class', // Enable dark mode support; use 'media' for OS-level preference
  theme: {
    extend: {
      colors: {
        customNavy: 'rgb(0, 0, 40)',
        customBlue: 'rgb(0, 152, 255)', // Your existing custom colors
        // Define additional colors or extend existing ones for dark mode here if needed
      },
      // Example of adding a dark color variant (extend within extend for specific scenarios)
      backgroundColor: {
        dark: { // Custom dark mode background colors
          primary: '#1E293B', // Darker shade for primary background in dark mode
          secondary: '#0f172a', // Darker shade for secondary background in dark mode
        }
      },
      textColor: {
        dark: { // Custom dark mode text colors
          primary: '#f0f6fc', // Lighter text for dark backgrounds
          secondary: '#94a3b8', // Another example text color for dark mode
        }
      }
    },
    container: {
      padding: "7rem",
      center: true,
    }
  },
  plugins: [require("@tailwindcss/typography")],
};

