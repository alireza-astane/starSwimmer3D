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
    sphere { m*<0.44318173407772626,1.0456499317627408,0.12872624285285258>, 1 }        
    sphere {  m*<0.6839168388194179,1.1743600099430662,3.1162810139734027>, 1 }
    sphere {  m*<3.1778901280839817,1.1476839071491152,-1.100483282598332>, 1 }
    sphere {  m*<-1.1784336258151629,3.3741238761813426,-0.8452195225631182>, 1}
    sphere { m*<-3.7821158872889127,-6.941669439949375,-2.31938453501799>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6839168388194179,1.1743600099430662,3.1162810139734027>, <0.44318173407772626,1.0456499317627408,0.12872624285285258>, 0.5 }
    cylinder { m*<3.1778901280839817,1.1476839071491152,-1.100483282598332>, <0.44318173407772626,1.0456499317627408,0.12872624285285258>, 0.5}
    cylinder { m*<-1.1784336258151629,3.3741238761813426,-0.8452195225631182>, <0.44318173407772626,1.0456499317627408,0.12872624285285258>, 0.5 }
    cylinder {  m*<-3.7821158872889127,-6.941669439949375,-2.31938453501799>, <0.44318173407772626,1.0456499317627408,0.12872624285285258>, 0.5}

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
    sphere { m*<0.44318173407772626,1.0456499317627408,0.12872624285285258>, 1 }        
    sphere {  m*<0.6839168388194179,1.1743600099430662,3.1162810139734027>, 1 }
    sphere {  m*<3.1778901280839817,1.1476839071491152,-1.100483282598332>, 1 }
    sphere {  m*<-1.1784336258151629,3.3741238761813426,-0.8452195225631182>, 1}
    sphere { m*<-3.7821158872889127,-6.941669439949375,-2.31938453501799>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6839168388194179,1.1743600099430662,3.1162810139734027>, <0.44318173407772626,1.0456499317627408,0.12872624285285258>, 0.5 }
    cylinder { m*<3.1778901280839817,1.1476839071491152,-1.100483282598332>, <0.44318173407772626,1.0456499317627408,0.12872624285285258>, 0.5}
    cylinder { m*<-1.1784336258151629,3.3741238761813426,-0.8452195225631182>, <0.44318173407772626,1.0456499317627408,0.12872624285285258>, 0.5 }
    cylinder {  m*<-3.7821158872889127,-6.941669439949375,-2.31938453501799>, <0.44318173407772626,1.0456499317627408,0.12872624285285258>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    