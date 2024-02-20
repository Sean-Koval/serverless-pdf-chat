/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        customNavy: 'rgb(0, 0, 40)',
        customBlue: 'rgb(0, 152, 255)', // Adding custom text/icon color
      },
    },
    container: {
      padding: "7rem",
      center: true,
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
