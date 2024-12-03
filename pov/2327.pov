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
    sphere { m*<1.0250152503165637,0.43189897971457347,0.47192449432735484>, 1 }        
    sphere {  m*<1.2689647825978958,0.4665347421710625,3.461787410502046>, 1 }
    sphere {  m*<3.7622119716604314,0.46653474217106233,-0.755494797988572>, 1 }
    sphere {  m*<-2.911305333398339,6.631911086811689,-1.8554815633622395>, 1}
    sphere { m*<-3.8028000849472323,-7.84688243804377,-2.381917629357323>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2689647825978958,0.4665347421710625,3.461787410502046>, <1.0250152503165637,0.43189897971457347,0.47192449432735484>, 0.5 }
    cylinder { m*<3.7622119716604314,0.46653474217106233,-0.755494797988572>, <1.0250152503165637,0.43189897971457347,0.47192449432735484>, 0.5}
    cylinder { m*<-2.911305333398339,6.631911086811689,-1.8554815633622395>, <1.0250152503165637,0.43189897971457347,0.47192449432735484>, 0.5 }
    cylinder {  m*<-3.8028000849472323,-7.84688243804377,-2.381917629357323>, <1.0250152503165637,0.43189897971457347,0.47192449432735484>, 0.5}

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
    sphere { m*<1.0250152503165637,0.43189897971457347,0.47192449432735484>, 1 }        
    sphere {  m*<1.2689647825978958,0.4665347421710625,3.461787410502046>, 1 }
    sphere {  m*<3.7622119716604314,0.46653474217106233,-0.755494797988572>, 1 }
    sphere {  m*<-2.911305333398339,6.631911086811689,-1.8554815633622395>, 1}
    sphere { m*<-3.8028000849472323,-7.84688243804377,-2.381917629357323>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2689647825978958,0.4665347421710625,3.461787410502046>, <1.0250152503165637,0.43189897971457347,0.47192449432735484>, 0.5 }
    cylinder { m*<3.7622119716604314,0.46653474217106233,-0.755494797988572>, <1.0250152503165637,0.43189897971457347,0.47192449432735484>, 0.5}
    cylinder { m*<-2.911305333398339,6.631911086811689,-1.8554815633622395>, <1.0250152503165637,0.43189897971457347,0.47192449432735484>, 0.5 }
    cylinder {  m*<-3.8028000849472323,-7.84688243804377,-2.381917629357323>, <1.0250152503165637,0.43189897971457347,0.47192449432735484>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    