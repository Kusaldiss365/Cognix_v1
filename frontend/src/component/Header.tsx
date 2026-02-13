import Link from "next/link";

export default function Header() {
  return (
    <header className="absolute px-4 pt-4">
        <Link href="/" className="flex">
          <img src="logo_cognix.png" alt="logo" className="h-15"/>
          <p className="text-2xl font-bold mt-[14px]">CogniX</p>
        </Link>
    </header>
  );
}
