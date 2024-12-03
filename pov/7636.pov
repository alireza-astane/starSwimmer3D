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
    sphere { m*<-0.5039820325245634,-0.6181146927231911,-0.4829518102469692>, 1 }        
    sphere {  m*<0.9151854616755987,0.3718242211567264,9.366338286788181>, 1 }
    sphere {  m*<8.282972659998402,0.08673197036446467,-5.204339142285753>, 1 }
    sphere {  m*<-6.612990533690597,6.609813343985101,-3.713532239104145>, 1}
    sphere { m*<-3.560019130581608,-7.2735667771640395,-1.898164540773195>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9151854616755987,0.3718242211567264,9.366338286788181>, <-0.5039820325245634,-0.6181146927231911,-0.4829518102469692>, 0.5 }
    cylinder { m*<8.282972659998402,0.08673197036446467,-5.204339142285753>, <-0.5039820325245634,-0.6181146927231911,-0.4829518102469692>, 0.5}
    cylinder { m*<-6.612990533690597,6.609813343985101,-3.713532239104145>, <-0.5039820325245634,-0.6181146927231911,-0.4829518102469692>, 0.5 }
    cylinder {  m*<-3.560019130581608,-7.2735667771640395,-1.898164540773195>, <-0.5039820325245634,-0.6181146927231911,-0.4829518102469692>, 0.5}

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
    sphere { m*<-0.5039820325245634,-0.6181146927231911,-0.4829518102469692>, 1 }        
    sphere {  m*<0.9151854616755987,0.3718242211567264,9.366338286788181>, 1 }
    sphere {  m*<8.282972659998402,0.08673197036446467,-5.204339142285753>, 1 }
    sphere {  m*<-6.612990533690597,6.609813343985101,-3.713532239104145>, 1}
    sphere { m*<-3.560019130581608,-7.2735667771640395,-1.898164540773195>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9151854616755987,0.3718242211567264,9.366338286788181>, <-0.5039820325245634,-0.6181146927231911,-0.4829518102469692>, 0.5 }
    cylinder { m*<8.282972659998402,0.08673197036446467,-5.204339142285753>, <-0.5039820325245634,-0.6181146927231911,-0.4829518102469692>, 0.5}
    cylinder { m*<-6.612990533690597,6.609813343985101,-3.713532239104145>, <-0.5039820325245634,-0.6181146927231911,-0.4829518102469692>, 0.5 }
    cylinder {  m*<-3.560019130581608,-7.2735667771640395,-1.898164540773195>, <-0.5039820325245634,-0.6181146927231911,-0.4829518102469692>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    