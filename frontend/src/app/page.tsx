'use client';
import { ArrowUpRight, CircleChevronRight } from 'lucide-react';
import { useRouter } from 'next/navigation';

export default function Home() {
  const router = useRouter();

  return (
    <div className="">
      <div className="flex flex-col items-center justify-center min-h-screen gap-8 font-sans">
        <h1 className="capitalize text-6xl font-semibold">Welcome to CogniX !</h1>
        <p className="text-xl -mt-4">Your AI Guide, Right by Your Side</p>
        <button 
        type="button"
        className="flex px-8 py-4 cursor-pointer bg-blue-600 rounded-xl hover:bg-blue-800 transition duration-300 ease-in-out"
        onClick={() => router.push('/upload')}
        ><p className='animate-pulse transition duration-300'>Get Started</p>
        <CircleChevronRight className='ms-2 mt-1 h-[18px] left-30 bottom-8 animate-pulse transition duration-300'/>
        </button>
      </div>
    </div>
  );
}
