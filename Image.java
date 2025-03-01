import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class Image
{
    public static double[] ImageToArray(String filepath) throws IOException
    {
        File imageFile = new File(filepath);
        BufferedImage img = ImageIO.read(imageFile);

        double[] result = new double[4096];
        int count = 0;

        for (int y = 0; y < img.getHeight(); y++) 
        {
            for (int x = 0; x < img.getWidth(); x++) 
            {
                //Retrieving contents of a pixel
                int pixel = img.getRGB(x,y);

                // Colours
                int r = (pixel >> 16) & 0xFF;
                int g = (pixel >> 8) & 0xFF;
                int b = (pixel & 0xFF);

                double grey = ((r + g + b) / 3);
                
                result[count] = grey;
                count++;
            }
        }

        return result;
    }
}