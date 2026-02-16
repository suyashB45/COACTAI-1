
import { Moon, Sun } from "lucide-react"
import { useTheme } from "./theme-provider"
import { Button } from "@/components/ui/button"

export function ModeToggle({ className }: { className?: string }) {
    const { setTheme, resolvedTheme } = useTheme()

    return (
        <Button
            variant="outline"
            size="icon"
            onClick={() => setTheme(resolvedTheme === "light" ? "dark" : "light")}
            className={`rounded-full w-10 h-10 transition-colors bg-secondary/50 dark:bg-muted/50 border-border/50 hover:bg-secondary/80 dark:hover:bg-muted/80 ${className}`}
            aria-label="Toggle theme"
        >
            {resolvedTheme === "dark" ? (
                <Moon className="h-[1.2rem] w-[1.2rem] transition-all text-primary" />
            ) : (
                <Sun className="h-[1.2rem] w-[1.2rem] transition-all text-primary" />
            )}
            <span className="sr-only">Toggle theme</span>
        </Button>
    )
}
