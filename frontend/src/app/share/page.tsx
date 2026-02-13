'use client';

import { useRouter, useSearchParams } from 'next/navigation';
import { CircleChevronRight } from 'lucide-react';
import ShareLinkBar from '@/component/ShareLinkBar';

export default function SharePage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const sessionId = searchParams.get('session_id');
  const userId = searchParams.get('user_id');


  return (
    <div className="min-h-screen flex flex-col w-2xl items-center justify-center">
      <div className="flex flex-col gap-10 p-10 rounded-lg bg-gray-900/50 w-full max-w-xl">
        <div className="flex flex-col items-center justify-center mb-6">
          <h1 className="text-2xl font-semibold mb-3">Ready to Start?</h1>
          <button
            type="button"
            onClick={() => router.push(`/chat?session_id=${sessionId}&user_id=${userId}`)}
            className="w-fit flex bg-blue-600 px-5 py-3 rounded-xl hover:bg-blue-800 transition duration-300 cursor-pointer"
          >
            Start Chat
            <CircleChevronRight className="h-4 mt-1 ml-2" />
          </button>
        </div>

        <div className="flex flex-col justify-center items-center">
          <h1 className="text-2xl font-semibold mb-2">Or Share this session</h1>
          <ShareLinkBar sessionId={sessionId} />
        </div>
      </div>
    </div>
  );
}
