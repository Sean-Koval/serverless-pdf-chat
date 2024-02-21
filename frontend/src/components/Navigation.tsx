import { Link } from "react-router-dom";
import { Menu } from "@headlessui/react";
import {
  ArrowLeftOnRectangleIcon,
  ChevronDownIcon,
} from "@heroicons/react/24/outline";
//import { ChatBubbleLeftRightIcon } from "@heroicons/react/24/solid";
import Symphony from "../../public/symphony.svg";

interface NavigationProps {
  userInfo: any;
  handleSignOutClick: (
    event: React.MouseEvent<HTMLButtonElement>
  ) => Promise<void>;
}

const Navigation: React.FC<NavigationProps> = ({
  userInfo,
  handleSignOutClick,
}: NavigationProps) => {
  return (
    <nav className="bg-customNavy">
      <div className="container flex flex-wrap items-center justify-between py-3">
        <Link
          to="/"
          className="inline-flex items-center self-center text-2xl font-semibold whitespace-nowrap text-customBlue"
        >
          <img src={Symphony} alt="React Logo" width={20} className="mr-1.5 py-2 mx-2" />
          <span>DocChat</span>
        </Link>
        <div className="absolute inset-y-0 right-0 flex items-center pr-2 sm:static sm:inset-auto sm:ml-6 sm:pr-0">
          <div className="relative ml-3">
            <Menu>
              <Menu.Button className="text-center inline-flex items-center text-white text-sm underline-offset-2 hover:underline">
                {userInfo?.attributes?.email}
                <ChevronDownIcon className="w-3 h-3 ml-1 text-customBlue" />
              </Menu.Button>
              <Menu.Items className="absolute right-0 z-10 mt-2 origin-top-right rounded-md bg-white py-1 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
                <div className="px-1 py-1 ">
                  <Menu.Item>
                    <button
                      onClick={handleSignOutClick}
                      className="group w-full inline-flex items-center rounded-md px-2 py-2 text-sm underline-offset-2 hover:underline"
                    >
                      <ArrowLeftOnRectangleIcon className="w-4 h-4 mr-1" />
                      Sign Out
                    </button>
                  </Menu.Item>
                </div>
              </Menu.Items>
            </Menu>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;

//          <ChatBubbleLeftRightIcon className="w-6 h-6 mr-1.5 text-customBlue" />

// import { Link } from "react-router-dom";
// import { Menu } from "@headlessui/react";
// import {
//   ArrowLeftOnRectangleIcon,
//   ChevronDownIcon,
// } from "@heroicons/react/24/outline";
// import { ChatBubbleLeftRightIcon } from "@heroicons/react/24/solid";
// //import Symphony from "../../public/symphony.svg";

// interface NavigationProps {
//   userInfo: any;
//   handleSignOutClick: (
//     event: React.MouseEvent<HTMLButtonElement>
//   ) => Promise<void>;
// }

// const Navigation: React.FC<NavigationProps> = ({
//   userInfo,
//   handleSignOutClick,
// }: NavigationProps) => {
//   return (
//     <nav className="bg-customNavy">
//       <div className="container flex flex-wrap items-center justify-between py-3">
//         <Link
//           to="/"
//           className="inline-flex items-center self-center text-2xl font-semibold whitespace-nowrap text-customBlue"
//         >
//           <ChatBubbleLeftRightIcon className="w-6 h-6 mr-1.5 text-customBlue" />           
//           Symphony DocChat
//         </Link>
//         <div className="absolute inset-y-0 right-0 flex items-center pr-2 sm:static sm:inset-auto sm:ml-6 sm:pr-0">
//           <div className="relative ml-3">
//             <Menu>
//               <Menu.Button className="text-center inline-flex items-center text-white text-sm underline-offset-2 hover:underline">
//                 {userInfo?.attributes?.email}
//                 <ChevronDownIcon className="w-3 h-3 ml-1 text-customBlue" />
//               </Menu.Button>
//               <Menu.Items className="absolute right-0 z-10 mt-2 origin-top-right rounded-md bg-white py-1 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
//                 <div className="px-1 py-1 ">
//                   <Menu.Item>
//                     <button
//                       onClick={handleSignOutClick}
//                       className="group w-full inline-flex items-center rounded-md px-2 py-2 text-sm underline-offset-2 hover:underline"
//                     >
//                       <ArrowLeftOnRectangleIcon className="w-4 h-4 mr-1" />
//                       Sign Out
//                     </button>
//                   </Menu.Item>
//                 </div>
//               </Menu.Items>
//             </Menu>
//           </div>
//         </div>
//       </div>
//     </nav>
//   );
// };

// export default Navigation;

//          <ChatBubbleLeftRightIcon className="w-6 h-6 mr-1.5 text-customBlue" />


// import React, { useState, useEffect } from "react";
// import { Link } from "react-router-dom";
// import { Menu } from "@headlessui/react";
// import {
//   ArrowLeftOnRectangleIcon,
//   ChevronDownIcon,
// } from "@heroicons/react/24/outline";
// import { ChatBubbleLeftRightIcon } from "@heroicons/react/24/solid";

// interface NavigationProps {
//   userInfo: any;
//   handleSignOutClick: (
//     event: React.MouseEvent<HTMLButtonElement>
//   ) => Promise<void>;
// }

// const Navigation: React.FC<NavigationProps> = ({
//   userInfo,
//   handleSignOutClick,
// }: NavigationProps) => {
//   // State to manage dark mode
//   const [darkMode, setDarkMode] = useState(() => {
//     // Initialize dark mode state based on user preference or system theme
//     return (
//       localStorage.getItem("darkMode") === "true" ||
//       window.matchMedia("(prefers-color-scheme: dark)").matches
//     );
//   });

//   // Effect to apply dark mode class to html element
//   useEffect(() => {
//     if (darkMode) {
//       document.documentElement.classList.add("dark");
//       localStorage.setItem("darkMode", "true");
//     } else {
//       document.documentElement.classList.remove("dark");
//       localStorage.setItem("darkMode", "false");
//     }
//   }, [darkMode]);

//   // Toggle dark mode
//   const toggleDarkMode = () => setDarkMode(!darkMode);

//   return (
//     <nav className="bg-customNavy">
//       <div className="container flex flex-wrap items-center justify-between py-3">
//         <Link
//           to="/"
//           className="inline-flex items-center self-center text-2xl font-semibold whitespace-nowrap text-customBlue"
//         >
//           <ChatBubbleLeftRightIcon className="w-6 h-6 mr-1.5 text-customBlue" />
//           Symphony DocChat
//         </Link>
//         <div className="flex items-center">
//           {/* Dark Mode Toggle Button */}
//           <button
//             onClick={toggleDarkMode}
//             className="mr-4 text-white hover:text-customBlue"
//           >
//             {darkMode ? 'Light Mode' : 'Dark Mode'}
//           </button>
//           <div className="relative">
//             <Menu>
//               <Menu.Button className="text-center inline-flex items-center text-white text-sm underline-offset-2 hover:underline">
//                 {userInfo?.attributes?.email}
//                 <ChevronDownIcon className="w-3 h-3 ml-1 text-customBlue" />
//               </Menu.Button>
//               <Menu.Items className="absolute right-0 z-10 mt-2 origin-top-right rounded-md bg-white py-1 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
//                 <div className="px-1 py-1 ">
//                   <Menu.Item>
//                     <button
//                       onClick={handleSignOutClick}
//                       className="group w-full inline-flex items-center rounded-md px-2 py-2 text-sm underline-offset-2 hover:underline"
//                     >
//                       <ArrowLeftOnRectangleIcon className="w-4 h-4 mr-1" />
//                       Sign Out
//                     </button>
//                   </Menu.Item>
//                 </div>
//               </Menu.Items>
//             </Menu>
//           </div>
//         </div>
//       </div>
//     </nav>
//   );
// };

// export default Navigation;