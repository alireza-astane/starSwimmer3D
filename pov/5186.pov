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
    sphere { m*<-0.4873362353594809,-0.14679917891493313,-1.5858831983997952>, 1 }        
    sphere {  m*<0.4429865523032945,0.2887864916201773,8.361208223596016>, 1 }
    sphere {  m*<3.4177710881645615,-0.0034370196350832216,-3.401680246643199>, 1 }
    sphere {  m*<-2.12016432139435,2.181805489364221,-2.5405916992384667>, 1}
    sphere { m*<-1.8523771003565186,-2.7058864530396765,-2.3510454140758963>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4429865523032945,0.2887864916201773,8.361208223596016>, <-0.4873362353594809,-0.14679917891493313,-1.5858831983997952>, 0.5 }
    cylinder { m*<3.4177710881645615,-0.0034370196350832216,-3.401680246643199>, <-0.4873362353594809,-0.14679917891493313,-1.5858831983997952>, 0.5}
    cylinder { m*<-2.12016432139435,2.181805489364221,-2.5405916992384667>, <-0.4873362353594809,-0.14679917891493313,-1.5858831983997952>, 0.5 }
    cylinder {  m*<-1.8523771003565186,-2.7058864530396765,-2.3510454140758963>, <-0.4873362353594809,-0.14679917891493313,-1.5858831983997952>, 0.5}

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
    sphere { m*<-0.4873362353594809,-0.14679917891493313,-1.5858831983997952>, 1 }        
    sphere {  m*<0.4429865523032945,0.2887864916201773,8.361208223596016>, 1 }
    sphere {  m*<3.4177710881645615,-0.0034370196350832216,-3.401680246643199>, 1 }
    sphere {  m*<-2.12016432139435,2.181805489364221,-2.5405916992384667>, 1}
    sphere { m*<-1.8523771003565186,-2.7058864530396765,-2.3510454140758963>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4429865523032945,0.2887864916201773,8.361208223596016>, <-0.4873362353594809,-0.14679917891493313,-1.5858831983997952>, 0.5 }
    cylinder { m*<3.4177710881645615,-0.0034370196350832216,-3.401680246643199>, <-0.4873362353594809,-0.14679917891493313,-1.5858831983997952>, 0.5}
    cylinder { m*<-2.12016432139435,2.181805489364221,-2.5405916992384667>, <-0.4873362353594809,-0.14679917891493313,-1.5858831983997952>, 0.5 }
    cylinder {  m*<-1.8523771003565186,-2.7058864530396765,-2.3510454140758963>, <-0.4873362353594809,-0.14679917891493313,-1.5858831983997952>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    