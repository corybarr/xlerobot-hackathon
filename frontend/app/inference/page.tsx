import { redirect } from "next/navigation";

/** Old Molmo-only page; inference is unified on the home wizard (Inference step). */
export default function LegacyInferencePage() {
  redirect("/");
}
