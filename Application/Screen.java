public class Screen {

	public int width;
	public int height;
	public int[] pixels;

	public Screen(int width, int height) {
		pixels = new int[width * height];
	}
	

	public void render() {
		for(int i = 0; i < this.width; i++){
			for(int j = 0; j < this.height; j++){
				
			}
		}
	}
	
	public void update() {
		
	}
	
	public void clear() {
		for (int i = 0; i < pixels.length; i++) {
			pixels[i] = 0;
		}
	}
}
