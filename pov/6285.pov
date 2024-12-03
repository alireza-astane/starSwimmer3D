#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-1.404261915644315,-0.4943434890861954,-0.9294531372212582>, 1 }        
    sphere {  m*<0.05107209439057003,0.09343573477184969,8.946688191304005>, 1 }
    sphere {  m*<7.406423532390547,0.004515458777492848,-5.632805098741349>, 1 }
    sphere {  m*<-4.3051512264373155,3.299700965389461,-2.416200005951132>, 1}
    sphere { m*<-2.744128748015229,-3.0957559762417386,-1.5895479507236951>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.05107209439057003,0.09343573477184969,8.946688191304005>, <-1.404261915644315,-0.4943434890861954,-0.9294531372212582>, 0.5 }
    cylinder { m*<7.406423532390547,0.004515458777492848,-5.632805098741349>, <-1.404261915644315,-0.4943434890861954,-0.9294531372212582>, 0.5}
    cylinder { m*<-4.3051512264373155,3.299700965389461,-2.416200005951132>, <-1.404261915644315,-0.4943434890861954,-0.9294531372212582>, 0.5 }
    cylinder {  m*<-2.744128748015229,-3.0957559762417386,-1.5895479507236951>, <-1.404261915644315,-0.4943434890861954,-0.9294531372212582>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-1.404261915644315,-0.4943434890861954,-0.9294531372212582>, 1 }        
    sphere {  m*<0.05107209439057003,0.09343573477184969,8.946688191304005>, 1 }
    sphere {  m*<7.406423532390547,0.004515458777492848,-5.632805098741349>, 1 }
    sphere {  m*<-4.3051512264373155,3.299700965389461,-2.416200005951132>, 1}
    sphere { m*<-2.744128748015229,-3.0957559762417386,-1.5895479507236951>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.05107209439057003,0.09343573477184969,8.946688191304005>, <-1.404261915644315,-0.4943434890861954,-0.9294531372212582>, 0.5 }
    cylinder { m*<7.406423532390547,0.004515458777492848,-5.632805098741349>, <-1.404261915644315,-0.4943434890861954,-0.9294531372212582>, 0.5}
    cylinder { m*<-4.3051512264373155,3.299700965389461,-2.416200005951132>, <-1.404261915644315,-0.4943434890861954,-0.9294531372212582>, 0.5 }
    cylinder {  m*<-2.744128748015229,-3.0957559762417386,-1.5895479507236951>, <-1.404261915644315,-0.4943434890861954,-0.9294531372212582>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    