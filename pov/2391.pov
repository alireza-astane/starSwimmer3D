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
    sphere { m*<0.9748356749686621,0.5124823720059286,0.4422550344542834>, 1 }        
    sphere {  m*<1.2186449975119293,0.5544801073316582,3.432034747019112>, 1 }
    sphere {  m*<3.711892186574467,0.554480107331658,-0.7852474614715066>, 1 }
    sphere {  m*<-2.753937819101624,6.333574258025302,-1.76243362267415>, 1}
    sphere { m*<-3.822640063157779,-7.790072175426992,-2.3936493843897475>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2186449975119293,0.5544801073316582,3.432034747019112>, <0.9748356749686621,0.5124823720059286,0.4422550344542834>, 0.5 }
    cylinder { m*<3.711892186574467,0.554480107331658,-0.7852474614715066>, <0.9748356749686621,0.5124823720059286,0.4422550344542834>, 0.5}
    cylinder { m*<-2.753937819101624,6.333574258025302,-1.76243362267415>, <0.9748356749686621,0.5124823720059286,0.4422550344542834>, 0.5 }
    cylinder {  m*<-3.822640063157779,-7.790072175426992,-2.3936493843897475>, <0.9748356749686621,0.5124823720059286,0.4422550344542834>, 0.5}

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
    sphere { m*<0.9748356749686621,0.5124823720059286,0.4422550344542834>, 1 }        
    sphere {  m*<1.2186449975119293,0.5544801073316582,3.432034747019112>, 1 }
    sphere {  m*<3.711892186574467,0.554480107331658,-0.7852474614715066>, 1 }
    sphere {  m*<-2.753937819101624,6.333574258025302,-1.76243362267415>, 1}
    sphere { m*<-3.822640063157779,-7.790072175426992,-2.3936493843897475>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2186449975119293,0.5544801073316582,3.432034747019112>, <0.9748356749686621,0.5124823720059286,0.4422550344542834>, 0.5 }
    cylinder { m*<3.711892186574467,0.554480107331658,-0.7852474614715066>, <0.9748356749686621,0.5124823720059286,0.4422550344542834>, 0.5}
    cylinder { m*<-2.753937819101624,6.333574258025302,-1.76243362267415>, <0.9748356749686621,0.5124823720059286,0.4422550344542834>, 0.5 }
    cylinder {  m*<-3.822640063157779,-7.790072175426992,-2.3936493843897475>, <0.9748356749686621,0.5124823720059286,0.4422550344542834>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    