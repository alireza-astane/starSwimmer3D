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
    sphere { m*<0.3371857209777217,0.8452796656596668,0.06731282175577705>, 1 }        
    sphere {  m*<0.5779208257194135,0.9739897438399925,3.0548675928763287>, 1 }
    sphere {  m*<3.0718941149839782,0.9473136410460413,-1.161896703695406>, 1 }
    sphere {  m*<-1.2844296389151681,3.1737536100782693,-0.906632943660192>, 1}
    sphere { m*<-3.436384578057611,-6.288113967391348,-2.1190699962038506>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5779208257194135,0.9739897438399925,3.0548675928763287>, <0.3371857209777217,0.8452796656596668,0.06731282175577705>, 0.5 }
    cylinder { m*<3.0718941149839782,0.9473136410460413,-1.161896703695406>, <0.3371857209777217,0.8452796656596668,0.06731282175577705>, 0.5}
    cylinder { m*<-1.2844296389151681,3.1737536100782693,-0.906632943660192>, <0.3371857209777217,0.8452796656596668,0.06731282175577705>, 0.5 }
    cylinder {  m*<-3.436384578057611,-6.288113967391348,-2.1190699962038506>, <0.3371857209777217,0.8452796656596668,0.06731282175577705>, 0.5}

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
    sphere { m*<0.3371857209777217,0.8452796656596668,0.06731282175577705>, 1 }        
    sphere {  m*<0.5779208257194135,0.9739897438399925,3.0548675928763287>, 1 }
    sphere {  m*<3.0718941149839782,0.9473136410460413,-1.161896703695406>, 1 }
    sphere {  m*<-1.2844296389151681,3.1737536100782693,-0.906632943660192>, 1}
    sphere { m*<-3.436384578057611,-6.288113967391348,-2.1190699962038506>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5779208257194135,0.9739897438399925,3.0548675928763287>, <0.3371857209777217,0.8452796656596668,0.06731282175577705>, 0.5 }
    cylinder { m*<3.0718941149839782,0.9473136410460413,-1.161896703695406>, <0.3371857209777217,0.8452796656596668,0.06731282175577705>, 0.5}
    cylinder { m*<-1.2844296389151681,3.1737536100782693,-0.906632943660192>, <0.3371857209777217,0.8452796656596668,0.06731282175577705>, 0.5 }
    cylinder {  m*<-3.436384578057611,-6.288113967391348,-2.1190699962038506>, <0.3371857209777217,0.8452796656596668,0.06731282175577705>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    