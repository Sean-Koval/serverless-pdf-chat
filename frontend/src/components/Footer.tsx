//import { CloudIcon } from "@heroicons/react/24/outline";
//import GitHub from "../../public/github.svg";
import Symphony from "../../public/symphony.svg";

const Footer: React.FC = () => {
  return (
    <div className="bg-gray-100 mt-auto">
      <footer className="container">
        <div className=" flex flex-row justify-between py-3 text-sm">
          <div className="inline-flex items-center">
            <img
              src={Symphony}
              alt="React Logo"
              width={20}
              className="mr-1.5 py-2 mx-2"
            />
            <span>Symphony Document Chat</span>
          </div>
          <div className="inline-flex items-center hover:underline underline-offset-2">
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Footer;
// import { CloudIcon } from "@heroicons/react/24/outline";
// //import GitHub from "../../public/github.svg";
// import Symphony from "../../public/symphony.svg";

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
//               src={Symphony}
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
