// import { CloudIcon } from "@heroicons/react/24/outline";
// import GitHub from "../../public/github.svg";

// const Footer: React.FC = () => {
//   return (
//     <div className="bg-gray-100 mt-auto">
//       <footer className="container">
//         <div className=" flex flex-row justify-between py-3 text-sm">
//           <div className="inline-flex items-center">
//             <CloudIcon className="w-5 h-5 mr-1.5" />
//             Symphony Document Chat - Powered by AWS
//           </div>
//           <div className="inline-flex items-center hover:underline underline-offset-2">
//             <img
//               src={GitHub}
//               alt="React Logo"
//               width={20}
//               className="mr-1.5 py-2 mx-2"
//             />
//           </div>
//         </div>
//       </footer>
//     </div>
//   );
// };

// export default Footer;

import { CloudIcon } from "@heroicons/react/24/outline";
import GitHub from "../../public/github.svg";

const Footer: React.FC = () => {
  return (
    <div className="bg-gray-100 dark:bg-gray-800 mt-auto">
      <footer className="container">
        <div className="flex flex-row justify-between py-3 text-sm text-gray-900 dark:text-gray-300">
          <div className="inline-flex items-center">
            <CloudIcon className="w-5 h-5 mr-1.5 text-gray-900 dark:text-gray-300" />
            Symphony Document Chat - Powered by AWS
          </div>
          <a href="https://github.com/yourusername/yourproject" className="inline-flex items-center hover:underline underline-offset-2 dark:hover:text-white">
            <img
              src={GitHub}
              alt="GitHub Logo"
              width={20}
              className="mr-1.5 py-2 mx-2 dark:invert"
            />
            GitHub
          </a>
        </div>
      </footer>
    </div>
  );
};

export default Footer;
