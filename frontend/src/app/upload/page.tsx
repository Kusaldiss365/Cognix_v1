'use client';

import { useRouter } from 'next/navigation';
import { UploadCloudIcon } from 'lucide-react';
import { useEffect, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { toast } from 'sonner';

export default function UploadPage() {
  const router = useRouter();
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

  const [notesFile, setNotesFile] = useState('');
  const [questionsFile, setQuestionsFile] = useState('');
  const [answersFile, setAnswersFile] = useState('');

  const [notesUploaded, setNotesUploaded] = useState(false);
  const [questionsUploaded, setQuestionsUploaded] = useState(false);
  const [answersUploaded, setAnswersUploaded] = useState(false);

  const [isLoading, setIsLoading] = useState(false);
  const [userId, setUserId] = useState('');

  useEffect(() => {
    let stored = localStorage.getItem('user_id');
    if (!stored) {
      stored = uuidv4();
      localStorage.setItem('user_id', stored);
    }
    setUserId(stored);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    const formData = new FormData();
    const sessionId = uuidv4();

    const notes = (document.getElementById("notes") as HTMLInputElement)
      .files?.[0];
    const questions = (document.getElementById("questions") as HTMLInputElement)
      .files?.[0];
    const answers = (document.getElementById("answers") as HTMLInputElement)
      .files?.[0];

    if (!notes || !questions) {
      toast.error("Please upload documents");
      setIsLoading(false);
      return;
    }

    formData.append("notes", notes);
    formData.append("questions", questions);
    if (answers) {
      formData.append("answers", answers);
    }

    try {
      const res = await fetch(`${baseUrl}/upload/${sessionId}`, {
        method: "POST",
        body: formData,
      });

      const result = await res.json();
      if (result.status === "success") {
        const chatRes = await fetch(`${baseUrl}/chat/${sessionId}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: userId,
            user_answer: "[START_SESSION]",
            question_index: 0,
          }),
        });

        if (chatRes.ok) {
          const chatData = await chatRes.json();
          if (chatData.index !== undefined && chatData.question) {
            router.replace(`/share?session_id=${sessionId}&user_id=${userId}`);
            return;
          } else {
            toast.error("Invalid session response.");
          }
        } else {
          toast.error("Failed to prepare your session.");
        }
      }
    } catch (err) {
      toast.error("Backend error. Check connection.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center text-white p-8 justify-center">
      <h1 className="text-3xl font-bold mb-6">Upload and Let Us Guide You!</h1>

      {isLoading && (
        <div className="flex flex-col fixed inset-0 bg-gray-950 bg-opacity-90 items-center justify-center z-50">
          <div className="relative w-16 h-16 mb-10">
            <div className="absolute inset-0 rounded-full border-4 border-blue-400 animate-ping [animation-duration:1.7s]" />
            <div className="absolute inset-0 rounded-full border-4 border-blue-600" />
          </div>
          <div className="text-white text-lg">Preparing your session...</div>
        </div>
      )}

      <form
        onSubmit={handleSubmit}
        encType="multipart/form-data"
        className="flex flex-col gap-4 w-full max-w-md bg-gray-900 p-10 rounded-xl"
      >
        {/* Notes */}
        <div className="mb-2">
          <span className="block mb-1">Upload Notes</span>
          <label
            className={`flex items-center justify-between border ${
              notesUploaded ? "border-green-500" : "border-gray-700"
            } rounded-md bg-gray-800 px-4 py-3 cursor-pointer`}
          >
            <div className="flex items-center space-x-2">
              <UploadCloudIcon className="text-blue-200 w-5 h-5" />
              <span className="text-gray-300 text-sm">
                {notesFile || "Upload"}
              </span>
            </div>
            <input
              type="file"
              name="notes"
              className="hidden"
              id="notes"
              onChange={(e) => {
                const file = e.target.files?.[0];
                setNotesFile(file?.name || "");
                setNotesUploaded(!!file);
              }}
            />
          </label>
        </div>

        {/* Questions */}
        <div className="mb-2">
          <span className="block mb-1">Upload Questions</span>
          <label
            className={`flex items-center justify-between border ${
              questionsUploaded ? "border-green-500" : "border-gray-700"
            } rounded-md bg-gray-800 px-4 py-3 cursor-pointer`}
          >
            <div className="flex items-center space-x-2">
              <UploadCloudIcon className="text-blue-200 w-5 h-5" />
              <span className="text-gray-300 text-sm">
                {questionsFile || "Upload"}
              </span>
            </div>
            <input
              type="file"
              name="questions"
              className="hidden"
              id="questions"
              onChange={(e) => {
                const file = e.target.files?.[0];
                setQuestionsFile(file?.name || "");
                setQuestionsUploaded(!!file);
              }}
            />
          </label>
        </div>

        {/* Answers */}
        <div className="mb-2">
          <span className="block mb-1">
            Upload Answers{" "}
            <span className="text-xs text-gray-400">(Optional)</span>
          </span>
          <label
            className={`flex items-center justify-between border ${
              answersUploaded ? "border-green-500" : "border-gray-700"
            } rounded-md bg-gray-800 px-4 py-3 cursor-pointer`}
          >
            <div className="flex items-center space-x-2">
              <UploadCloudIcon className="text-blue-200 w-5 h-5" />
              <span className="text-gray-300 text-sm">
                {answersFile || "Upload"}
              </span>
            </div>
            <input
              type="file"
              name="answers"
              className="hidden"
              id="answers"
              onChange={(e) => {
                const file = e.target.files?.[0];
                setAnswersFile(file?.name || "");
                setAnswersUploaded(!!file);
              }}
            />
          </label>
        </div>

        {/* Buttons */}
        <div className="flex justify-between mt-5">
          <button
            type="button"
            className="border border-white px-6 rounded cursor-pointer transition duration-300 hover:border-gray-400 hover:text-gray-400"
            onClick={() => router.push("/")}
          >
            Back
          </button>
          <button
            type="submit"
            className="bg-blue-600 px-6 py-3 rounded hover:bg-blue-900 transition duration-300 cursor-pointer"
          >
            Generate
          </button>
        </div>
      </form>
    </div>
  );
}
