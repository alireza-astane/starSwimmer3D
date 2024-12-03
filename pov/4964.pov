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
    sphere { m*<-0.2627097960187016,-0.13567674223240345,-1.615964506024226>, 1 }        
    sphere {  m*<0.5200805195663649,0.2828455286506204,8.098567758069425>, 1 }
    sphere {  m*<2.471998597987555,-0.033642766846029355,-2.8451740314754095>, 1 }
    sphere {  m*<-1.8843251559115917,2.1927972021861954,-2.589910271440196>, 1}
    sphere { m*<-1.6165379348737599,-2.694894740217702,-2.400363986277623>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5200805195663649,0.2828455286506204,8.098567758069425>, <-0.2627097960187016,-0.13567674223240345,-1.615964506024226>, 0.5 }
    cylinder { m*<2.471998597987555,-0.033642766846029355,-2.8451740314754095>, <-0.2627097960187016,-0.13567674223240345,-1.615964506024226>, 0.5}
    cylinder { m*<-1.8843251559115917,2.1927972021861954,-2.589910271440196>, <-0.2627097960187016,-0.13567674223240345,-1.615964506024226>, 0.5 }
    cylinder {  m*<-1.6165379348737599,-2.694894740217702,-2.400363986277623>, <-0.2627097960187016,-0.13567674223240345,-1.615964506024226>, 0.5}

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
    sphere { m*<-0.2627097960187016,-0.13567674223240345,-1.615964506024226>, 1 }        
    sphere {  m*<0.5200805195663649,0.2828455286506204,8.098567758069425>, 1 }
    sphere {  m*<2.471998597987555,-0.033642766846029355,-2.8451740314754095>, 1 }
    sphere {  m*<-1.8843251559115917,2.1927972021861954,-2.589910271440196>, 1}
    sphere { m*<-1.6165379348737599,-2.694894740217702,-2.400363986277623>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5200805195663649,0.2828455286506204,8.098567758069425>, <-0.2627097960187016,-0.13567674223240345,-1.615964506024226>, 0.5 }
    cylinder { m*<2.471998597987555,-0.033642766846029355,-2.8451740314754095>, <-0.2627097960187016,-0.13567674223240345,-1.615964506024226>, 0.5}
    cylinder { m*<-1.8843251559115917,2.1927972021861954,-2.589910271440196>, <-0.2627097960187016,-0.13567674223240345,-1.615964506024226>, 0.5 }
    cylinder {  m*<-1.6165379348737599,-2.694894740217702,-2.400363986277623>, <-0.2627097960187016,-0.13567674223240345,-1.615964506024226>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    