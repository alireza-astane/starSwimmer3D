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
    sphere { m*<-0.7961261114190183,-1.254347453140701,-0.6182400958054624>, 1 }        
    sphere {  m*<0.6230413827811448,-0.2644085392607829,9.231050001229695>, 1 }
    sphere {  m*<7.990828581103953,-0.5495007900530449,-5.339627427844247>, 1 }
    sphere {  m*<-6.905134612585051,5.973580583567612,-3.8488205246626404>, 1}
    sphere { m*<-2.085487788263839,-4.062325427364502,-1.215327434988467>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6230413827811448,-0.2644085392607829,9.231050001229695>, <-0.7961261114190183,-1.254347453140701,-0.6182400958054624>, 0.5 }
    cylinder { m*<7.990828581103953,-0.5495007900530449,-5.339627427844247>, <-0.7961261114190183,-1.254347453140701,-0.6182400958054624>, 0.5}
    cylinder { m*<-6.905134612585051,5.973580583567612,-3.8488205246626404>, <-0.7961261114190183,-1.254347453140701,-0.6182400958054624>, 0.5 }
    cylinder {  m*<-2.085487788263839,-4.062325427364502,-1.215327434988467>, <-0.7961261114190183,-1.254347453140701,-0.6182400958054624>, 0.5}

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
    sphere { m*<-0.7961261114190183,-1.254347453140701,-0.6182400958054624>, 1 }        
    sphere {  m*<0.6230413827811448,-0.2644085392607829,9.231050001229695>, 1 }
    sphere {  m*<7.990828581103953,-0.5495007900530449,-5.339627427844247>, 1 }
    sphere {  m*<-6.905134612585051,5.973580583567612,-3.8488205246626404>, 1}
    sphere { m*<-2.085487788263839,-4.062325427364502,-1.215327434988467>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6230413827811448,-0.2644085392607829,9.231050001229695>, <-0.7961261114190183,-1.254347453140701,-0.6182400958054624>, 0.5 }
    cylinder { m*<7.990828581103953,-0.5495007900530449,-5.339627427844247>, <-0.7961261114190183,-1.254347453140701,-0.6182400958054624>, 0.5}
    cylinder { m*<-6.905134612585051,5.973580583567612,-3.8488205246626404>, <-0.7961261114190183,-1.254347453140701,-0.6182400958054624>, 0.5 }
    cylinder {  m*<-2.085487788263839,-4.062325427364502,-1.215327434988467>, <-0.7961261114190183,-1.254347453140701,-0.6182400958054624>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    