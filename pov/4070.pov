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
    sphere { m*<-0.15610891199615243,-0.07868211267326095,-0.2930333115523624>, 1 }        
    sphere {  m*<0.12450797509774597,0.07135093512560481,3.1894596999838263>, 1 }
    sphere {  m*<2.5785994820101044,0.023351862713113153,-1.5222428370035461>, 1 }
    sphere {  m*<-1.7777242718890427,2.2497918317453376,-1.266979076968333>, 1}
    sphere { m*<-1.5099370508512109,-2.63790011065856,-1.0774327918057602>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.12450797509774597,0.07135093512560481,3.1894596999838263>, <-0.15610891199615243,-0.07868211267326095,-0.2930333115523624>, 0.5 }
    cylinder { m*<2.5785994820101044,0.023351862713113153,-1.5222428370035461>, <-0.15610891199615243,-0.07868211267326095,-0.2930333115523624>, 0.5}
    cylinder { m*<-1.7777242718890427,2.2497918317453376,-1.266979076968333>, <-0.15610891199615243,-0.07868211267326095,-0.2930333115523624>, 0.5 }
    cylinder {  m*<-1.5099370508512109,-2.63790011065856,-1.0774327918057602>, <-0.15610891199615243,-0.07868211267326095,-0.2930333115523624>, 0.5}

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
    sphere { m*<-0.15610891199615243,-0.07868211267326095,-0.2930333115523624>, 1 }        
    sphere {  m*<0.12450797509774597,0.07135093512560481,3.1894596999838263>, 1 }
    sphere {  m*<2.5785994820101044,0.023351862713113153,-1.5222428370035461>, 1 }
    sphere {  m*<-1.7777242718890427,2.2497918317453376,-1.266979076968333>, 1}
    sphere { m*<-1.5099370508512109,-2.63790011065856,-1.0774327918057602>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.12450797509774597,0.07135093512560481,3.1894596999838263>, <-0.15610891199615243,-0.07868211267326095,-0.2930333115523624>, 0.5 }
    cylinder { m*<2.5785994820101044,0.023351862713113153,-1.5222428370035461>, <-0.15610891199615243,-0.07868211267326095,-0.2930333115523624>, 0.5}
    cylinder { m*<-1.7777242718890427,2.2497918317453376,-1.266979076968333>, <-0.15610891199615243,-0.07868211267326095,-0.2930333115523624>, 0.5 }
    cylinder {  m*<-1.5099370508512109,-2.63790011065856,-1.0774327918057602>, <-0.15610891199615243,-0.07868211267326095,-0.2930333115523624>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    