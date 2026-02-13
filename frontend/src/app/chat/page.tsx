'use client';

import {
  ArrowBigRightDash,
  Ban,
  CircleEllipsis,
  Lightbulb,
  SendHorizonal,
} from "lucide-react";
import { useState, useEffect, useRef } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { v4 as uuidv4 } from "uuid";

interface Message {
  sender: "user" | "bot";
  text: string;
}

export default function ChatPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const sessionId = searchParams.get("session_id");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [questionIndex, setQuestionIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [isInitializing, setIsInitializing] = useState(true);
  const [showOptions, setShowOptions] = useState(false);
  const [showEndModal, setShowEndModal] = useState(false);
  const [userId, setUserId] = useState("");
  const [isFinalEvaluationShown, setIsFinalEvaluationShown] = useState(false);
  const [totalQuestions, setTotalQuestions] = useState<number | null>(null);
  const optionsRef = useRef<HTMLDivElement | null>(null);
  const [chatComplete, setChatComplete] = useState(false);

  useEffect(() => {
    let id = searchParams.get("user_id") || localStorage.getItem("user_id");
    if (!id) id = uuidv4();
    localStorage.setItem("user_id", id);
    const url = new URL(window.location.href);
    url.searchParams.set("user_id", id);
    window.history.replaceState(null, "", url.toString());
    setUserId(id);
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        optionsRef.current &&
        !optionsRef.current.contains(event.target as Node)
      ) {
        setShowOptions(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const autoResize = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      const scrollHeight = textareaRef.current.scrollHeight;
      const maxHeight = 5 * 28;
      textareaRef.current.style.overflowY =
        scrollHeight > maxHeight ? "auto" : "hidden";
      textareaRef.current.style.height = `${Math.min(
        scrollHeight,
        maxHeight
      )}px`;
    }
  };

  useEffect(() => autoResize(), [input]);
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (container) container.scrollTop = container.scrollHeight;
  }, [messages, isLoading, isInitializing, showEndModal]);

  useEffect(() => {
    if (!userId || !sessionId) return;
    setMessages([
      {
        sender: "bot",
        text: `👋 Welcome to CogniX! Let's improve your learning with interactive feedback. Hold on while we get your first question...`,
      },
    ]);
    fetchFirstQuestion();
  }, [userId, sessionId]);

  const fetchFirstQuestion = async () => {
    setIsInitializing(true);
    try {
      const res = await fetch(`${baseUrl}/chat/${sessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          user_answer: "[START_SESSION]",
          question_index: 0,
        }),
      });
      const data = await res.json();
      setTotalQuestions(data.total_questions);
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: `🤔 Question ${data.index + 1} of ${data.total_questions}:<br>${
            data.question
          }`,
        },
      ]);
      setQuestionIndex(data.index);
    } catch {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "⚠️ Failed to load the first question." },
      ]);
    } finally {
      setIsInitializing(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isLoading || isInitializing) return;
    setMessages((prev) => [...prev, { sender: "user", text: trimmed }]);
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch(`${baseUrl}/chat/${sessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          user_answer: trimmed,
          question_index: questionIndex,
        }),
      });
      const data = await res.json();

      const botMessages: Message[] = [{ sender: "bot", text: data.message }];

      if (data.retry) {
        botMessages.push({
          sender: "bot",
          text: `🔁 Let's try answering the question again`,
        });
      } else if (data.question) {
        botMessages.push({
          sender: "bot",
          text: `🤔 Question ${data.index + 1} of ${data.total_questions}:<br>${
            data.question
          }`,
        });
      }

      if (data.complete) {
        if (data.final_summary) {
          botMessages.push({
            sender: "bot",
            text: `🎉 Final Evaluation:<br>${data.final_summary}`,
          });
          setChatComplete(true);
        } else {
          botMessages.push({
            sender: "bot",
            text: "🎉 All questions completed!",
          });
        }
      }

      setMessages((prev) => [...prev, ...botMessages]);
      setQuestionIndex(data.index);
    } catch {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "⚠️ Something went wrong. Please try again." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // useEffect(() => {
  //   if (chatComplete) {
  //     const timer = setTimeout(() => {
  //       router.push("/");
  //     }, 30000);

  //     return () => clearTimeout(timer);
  //   }
  // }, [chatComplete]);

  const handleNextQuestion = async () => {
    try {
      const res = await fetch(`${baseUrl}/chat/${sessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          user_answer: "[NEXT_QUESTION]",
          question_index: questionIndex,
        }),
      });
      const data = await res.json();
      if (data.complete) return router.push("/");
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: `🤔 ${data.message}` },
      ]);
      setQuestionIndex(data.index);
    } catch {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "⚠️ Failed to move to next question." },
      ]);
    }
    setShowOptions(false);
  };

  const handleEndChat = async () => {
    try {
      const res = await fetch(`${baseUrl}/chat/${sessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          user_answer: "[END_CHAT]",
          question_index: questionIndex,
        }),
      });
      const data = await res.json();
      setMessages((prev) => [...prev, { sender: "bot", text: data.message }]);
      if (data.complete) setShowEndModal(true);
    } catch {
      console.error("End chat error");
    }
    setShowOptions(false);
  };

  const handleGetHint = async () => {
    if (!userId || !sessionId || isLoading || isInitializing) return;

    setIsLoading(true);

    try {
      const res = await fetch(`${baseUrl}/chat/${sessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          user_answer: "[GET_HINT_ONLY]",
          question_index: questionIndex,
        }),
      });

      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: `💡 ${data.message}` },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "⚠️ Couldn’t fetch help at the moment." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // useEffect(() => {
  //   const navEntry = performance.getEntriesByType(
  //     "navigation"
  //   )[0] as PerformanceNavigationTiming;

  //   if (navEntry?.type === "reload") {
  //     sessionStorage.clear();
  //     localStorage.clear();
  //     router.replace("/");
  //   }
  // }, []);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center text-white p-6">
      {showEndModal && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-gray-950">
          <div className="flex flex-col rounded-lg shadow-lg p-6 max-w-sm text-center bg-gray-800 text-white justify-center items-center">
            <img src="/fireworks.png" alt="User" className="w-20 h-20 mb-6" />
            <h2 className="text-xl font-semibold mb-6">
              Thanks for learning with CogniX!
            </h2>
            <p className="mb-10 text-sm">
              {" "}
              You've reached the end. Great job! Feel free to return anytime to
              challenge yourself, review concepts and continue improving your
              understanding.
            </p>
            <button
              onClick={() => router.push("/")}
              className="bg-blue-600 text-white px-4 py-1 rounded-2xl hover:bg-blue-700 transition cursor-pointer"
            >
              Close
            </button>
          </div>
        </div>
      )}

      <div className="w-4xl relative flex flex-col bg-black/60 rounded-xl shadow-xl h-[75vh] overflow-hidden">
        <div className="absolute top-3 right-5 z-20">
          <div title="Need help?">
            <Lightbulb
              onClick={isFinalEvaluationShown ? undefined : handleGetHint}
              className={`w-6 h-6 cursor-pointer ${
                isFinalEvaluationShown
                  ? "text-gray-600 cursor-not-allowed"
                  : "text-yellow-500 hover:text-yellow-300"
              }`}
            />
          </div>
        </div>
        <div
          ref={scrollContainerRef}
          className="flex-1 overflow-y-auto p-4 flex flex-col gap-4 scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent"
        >
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${
                msg.sender === "user" ? "justify-end" : "justify-start"
              } gap-2`}
            >
              {msg.sender === "bot" && (
                <img
                  src="/logo_cognix.png"
                  alt="Cognix"
                  className="w-6 h-6 rounded-full border border-blue-900 p-1"
                />
              )}
              <div
                className={`px-4 py-2 rounded-xl max-w-[75%] break-words text-sm ${
                  msg.sender === "user" ? "bg-[#333]" : "bg-gray-800"
                }`}
                dangerouslySetInnerHTML={{ __html: msg.text }}
              />
              {msg.sender === "user" && (
                <img
                  src="/user.png"
                  alt="User"
                  className="w-6 h-6 rounded-full border border-blue-900 p-1"
                />
              )}
            </div>
          ))}

          {(isLoading || isInitializing) && (
            <div className="flex items-start gap-2 mb-2">
              <img
                src="/logo_cognix.png"
                alt="Cognix"
                className="w-6 h-6 rounded-full border border-blue-900 p-1"
              />
              <div className="bg-gray-800 px-4 py-2 rounded-xl max-w-[75%] text-sm flex gap-1">
                <span className="w-1 h-1 bg-white rounded-full animate-bounce [animation-delay:0ms]" />
                <span className="w-1 h-1 bg-white rounded-full animate-bounce [animation-delay:150ms]" />
                <span className="w-1 h-1 bg-white rounded-full animate-bounce [animation-delay:300ms]" />
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        <form
          onSubmit={handleSubmit}
          className="border-t border-gray-800 p-3 flex items-center bg-black/30 relative"
        >
          <div ref={optionsRef} className="relative">
            <CircleEllipsis
              onClick={() => setShowOptions((prev) => !prev)}
              className="text-blue-400 hover:text-blue-600 w-5 h-5 cursor-pointer me-2"
            />
            {showOptions && (
              <div className="absolute bottom-10 left-0 bg-gray-900 border border-gray-700 rounded-md shadow-lg z-10 w-48">
                <button
                  onClick={handleNextQuestion}
                  disabled={
                    !totalQuestions || questionIndex + 1 >= totalQuestions
                  }
                  className={`flex w-full items-center px-4 py-2 text-sm text-left cursor-pointer ${
                    !totalQuestions || questionIndex + 1 >= totalQuestions
                      ? "opacity-50 cursor-not-allowed"
                      : "hover:bg-gray-700"
                  }`}
                >
                  <ArrowBigRightDash className="w-5 h-5 mr-2" /> Next Question
                </button>

                <button
                  onClick={handleEndChat}
                  className="flex w-full items-center px-4 py-2 text-sm text-left hover:bg-gray-700 cursor-pointer"
                >
                  <Ban className="w-4 h-4 mr-2" /> End Chat
                </button>
              </div>
            )}
          </div>
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onInput={autoResize}
            disabled={isLoading || isInitializing}
            placeholder="Write your answer..."
            rows={1}
            className="bg-transparent flex-1 text-white focus:outline-none placeholder-gray-400 text-sm disabled:opacity-50 resize-none leading-[1.8rem] py-2"
          />
          <button type="submit" disabled={isLoading || isInitializing}>
            <SendHorizonal className="text-blue-400 hover:text-blue-600 w-5 h-5 cursor-pointer" />
          </button>
        </form>
      </div>
    </div>
  );
}
