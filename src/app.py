if __package__:
	from .image_classifier import main
else:
	from image_classifier import main


if __name__ == "__main__":
	main()
