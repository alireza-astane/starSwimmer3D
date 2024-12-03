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
    sphere { m*<1.0393534856511695,-9.264043305475092e-19,0.7584775538099224>, 1 }        
    sphere {  m*<1.2213038913942473,1.1337864937338033e-18,3.75296199264403>, 1 }
    sphere {  m*<5.181279307573439,5.39725125135371e-18,-0.9955713534445378>, 1 }
    sphere {  m*<-3.8681919715237862,8.164965809277259,-2.2828198988044788>, 1}
    sphere { m*<-3.8681919715237862,-8.164965809277259,-2.2828198988044823>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2213038913942473,1.1337864937338033e-18,3.75296199264403>, <1.0393534856511695,-9.264043305475092e-19,0.7584775538099224>, 0.5 }
    cylinder { m*<5.181279307573439,5.39725125135371e-18,-0.9955713534445378>, <1.0393534856511695,-9.264043305475092e-19,0.7584775538099224>, 0.5}
    cylinder { m*<-3.8681919715237862,8.164965809277259,-2.2828198988044788>, <1.0393534856511695,-9.264043305475092e-19,0.7584775538099224>, 0.5 }
    cylinder {  m*<-3.8681919715237862,-8.164965809277259,-2.2828198988044823>, <1.0393534856511695,-9.264043305475092e-19,0.7584775538099224>, 0.5}

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
    sphere { m*<1.0393534856511695,-9.264043305475092e-19,0.7584775538099224>, 1 }        
    sphere {  m*<1.2213038913942473,1.1337864937338033e-18,3.75296199264403>, 1 }
    sphere {  m*<5.181279307573439,5.39725125135371e-18,-0.9955713534445378>, 1 }
    sphere {  m*<-3.8681919715237862,8.164965809277259,-2.2828198988044788>, 1}
    sphere { m*<-3.8681919715237862,-8.164965809277259,-2.2828198988044823>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2213038913942473,1.1337864937338033e-18,3.75296199264403>, <1.0393534856511695,-9.264043305475092e-19,0.7584775538099224>, 0.5 }
    cylinder { m*<5.181279307573439,5.39725125135371e-18,-0.9955713534445378>, <1.0393534856511695,-9.264043305475092e-19,0.7584775538099224>, 0.5}
    cylinder { m*<-3.8681919715237862,8.164965809277259,-2.2828198988044788>, <1.0393534856511695,-9.264043305475092e-19,0.7584775538099224>, 0.5 }
    cylinder {  m*<-3.8681919715237862,-8.164965809277259,-2.2828198988044823>, <1.0393534856511695,-9.264043305475092e-19,0.7584775538099224>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    