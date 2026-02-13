'use client';

import { useRef } from "react";
import { Copy } from "lucide-react";
import { toast } from "sonner";

export default function ShareLinkBar({
  sessionId,
}: {
  sessionId: string | null;
}) {
  const inputRef = useRef<HTMLInputElement>(null);

  const origin =
    process.env.NEXT_PUBLIC_FRONTEND_ORIGIN ||
    (typeof window !== "undefined" ? window.location.origin : "");
  const url = new URL(origin + "/chat");
  if (sessionId) url.searchParams.set("session_id", sessionId);
  const shareUrl = url.toString();

  const handleCopy = async () => {
    const inputEl = inputRef.current;
    if (!inputEl) {
      toast.error("Input field not found.");
      return;
    }

    if (!navigator.clipboard || !window.isSecureContext) {
      // toast.error("Clipboard not available in this environment.");
      return;
    }

    try {
      await navigator.clipboard.writeText(inputEl.value);
      toast.success("Link copied to clipboard");
    } catch (err) {
      console.error("Clipboard copy failed:", err);
      toast.error("Failed to copy link");
    }
  };

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="flex items-center bg-gray-900 border border-gray-700 rounded-full overflow-hidden px-4 py-2 w-fit max-w-full">
        <input
          ref={inputRef}
          type="text"
          readOnly
          value={shareUrl}
          className="bg-transparent text-white text-sm truncate focus:outline-none w-[250px]"
        />

        <button
          type="button"
          onClick={handleCopy}
          className="group text-sm text-white rounded-md px-3 py-1.5 cursor-pointer transition"
        >
          <Copy className="w-4 h-4 text-white group-hover:text-gray-400 transition" />
        </button>
      </div>
    </div>
  );
}
