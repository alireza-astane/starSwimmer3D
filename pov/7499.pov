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
    sphere { m*<-0.5727606052218672,-0.7679009945040705,-0.5148023109266016>, 1 }        
    sphere {  m*<0.8464068889782946,0.22203791937584683,9.334487786108548>, 1 }
    sphere {  m*<8.214194087301097,-0.06305433141641448,-5.236189642965383>, 1 }
    sphere {  m*<-6.681769106387893,6.460027042204222,-3.745382739783776>, 1}
    sphere { m*<-3.235884500833988,-6.567664843393398,-1.7480618294170764>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8464068889782946,0.22203791937584683,9.334487786108548>, <-0.5727606052218672,-0.7679009945040705,-0.5148023109266016>, 0.5 }
    cylinder { m*<8.214194087301097,-0.06305433141641448,-5.236189642965383>, <-0.5727606052218672,-0.7679009945040705,-0.5148023109266016>, 0.5}
    cylinder { m*<-6.681769106387893,6.460027042204222,-3.745382739783776>, <-0.5727606052218672,-0.7679009945040705,-0.5148023109266016>, 0.5 }
    cylinder {  m*<-3.235884500833988,-6.567664843393398,-1.7480618294170764>, <-0.5727606052218672,-0.7679009945040705,-0.5148023109266016>, 0.5}

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
    sphere { m*<-0.5727606052218672,-0.7679009945040705,-0.5148023109266016>, 1 }        
    sphere {  m*<0.8464068889782946,0.22203791937584683,9.334487786108548>, 1 }
    sphere {  m*<8.214194087301097,-0.06305433141641448,-5.236189642965383>, 1 }
    sphere {  m*<-6.681769106387893,6.460027042204222,-3.745382739783776>, 1}
    sphere { m*<-3.235884500833988,-6.567664843393398,-1.7480618294170764>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8464068889782946,0.22203791937584683,9.334487786108548>, <-0.5727606052218672,-0.7679009945040705,-0.5148023109266016>, 0.5 }
    cylinder { m*<8.214194087301097,-0.06305433141641448,-5.236189642965383>, <-0.5727606052218672,-0.7679009945040705,-0.5148023109266016>, 0.5}
    cylinder { m*<-6.681769106387893,6.460027042204222,-3.745382739783776>, <-0.5727606052218672,-0.7679009945040705,-0.5148023109266016>, 0.5 }
    cylinder {  m*<-3.235884500833988,-6.567664843393398,-1.7480618294170764>, <-0.5727606052218672,-0.7679009945040705,-0.5148023109266016>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    