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
    sphere { m*<-0.26203706660358034,-0.13531706450402492,-1.6076158442203103>, 1 }        
    sphere {  m*<0.5179446727594996,0.28170358880022844,8.072061614210266>, 1 }
    sphere {  m*<2.4726713274026766,-0.033283089117650815,-2.8368253696714936>, 1 }
    sphere {  m*<-1.8836524264964705,2.193156879914574,-2.58156160963628>, 1}
    sphere { m*<-1.6158652054586387,-2.694535062489323,-2.3920153244737072>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5179446727594996,0.28170358880022844,8.072061614210266>, <-0.26203706660358034,-0.13531706450402492,-1.6076158442203103>, 0.5 }
    cylinder { m*<2.4726713274026766,-0.033283089117650815,-2.8368253696714936>, <-0.26203706660358034,-0.13531706450402492,-1.6076158442203103>, 0.5}
    cylinder { m*<-1.8836524264964705,2.193156879914574,-2.58156160963628>, <-0.26203706660358034,-0.13531706450402492,-1.6076158442203103>, 0.5 }
    cylinder {  m*<-1.6158652054586387,-2.694535062489323,-2.3920153244737072>, <-0.26203706660358034,-0.13531706450402492,-1.6076158442203103>, 0.5}

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
    sphere { m*<-0.26203706660358034,-0.13531706450402492,-1.6076158442203103>, 1 }        
    sphere {  m*<0.5179446727594996,0.28170358880022844,8.072061614210266>, 1 }
    sphere {  m*<2.4726713274026766,-0.033283089117650815,-2.8368253696714936>, 1 }
    sphere {  m*<-1.8836524264964705,2.193156879914574,-2.58156160963628>, 1}
    sphere { m*<-1.6158652054586387,-2.694535062489323,-2.3920153244737072>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5179446727594996,0.28170358880022844,8.072061614210266>, <-0.26203706660358034,-0.13531706450402492,-1.6076158442203103>, 0.5 }
    cylinder { m*<2.4726713274026766,-0.033283089117650815,-2.8368253696714936>, <-0.26203706660358034,-0.13531706450402492,-1.6076158442203103>, 0.5}
    cylinder { m*<-1.8836524264964705,2.193156879914574,-2.58156160963628>, <-0.26203706660358034,-0.13531706450402492,-1.6076158442203103>, 0.5 }
    cylinder {  m*<-1.6158652054586387,-2.694535062489323,-2.3920153244737072>, <-0.26203706660358034,-0.13531706450402492,-1.6076158442203103>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    