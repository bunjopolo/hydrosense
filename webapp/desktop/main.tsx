import { createRoot } from "react-dom/client";
import Dashboard from "../app/page";
import "../app/globals.css";
import "./desktop.css";

createRoot(document.getElementById("root")!).render(<Dashboard/>);
