package main

import "os"

func main() {
	switch os.Args[1] {
	case "serve":
		serve(os.Getenv("LYCAON_ADDR"), os.Getenv("LYCAON_CONFIG_DIR"), os.Getenv("LYCAON_API_TOKEN"))
	case "diagnostics":
		if os.Args[2] == "document-core" {
			documentCore()
		}
	}
}
