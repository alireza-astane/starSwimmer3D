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
    sphere { m*<-0.27070624899062296,-0.13833207879751022,-1.6733138018708225>, 1 }        
    sphere {  m*<0.5336723745453924,0.290800793971149,8.285042123638904>, 1 }
    sphere {  m*<2.4829071619296808,-0.035617462703209646,-2.911660000864776>, 1 }
    sphere {  m*<-1.8924924311002345,2.190143694416979,-2.6469707017386166>, 1}
    sphere { m*<-1.6247052100624026,-2.6975482479869184,-2.457424416576046>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5336723745453924,0.290800793971149,8.285042123638904>, <-0.27070624899062296,-0.13833207879751022,-1.6733138018708225>, 0.5 }
    cylinder { m*<2.4829071619296808,-0.035617462703209646,-2.911660000864776>, <-0.27070624899062296,-0.13833207879751022,-1.6733138018708225>, 0.5}
    cylinder { m*<-1.8924924311002345,2.190143694416979,-2.6469707017386166>, <-0.27070624899062296,-0.13833207879751022,-1.6733138018708225>, 0.5 }
    cylinder {  m*<-1.6247052100624026,-2.6975482479869184,-2.457424416576046>, <-0.27070624899062296,-0.13833207879751022,-1.6733138018708225>, 0.5}

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
    sphere { m*<-0.27070624899062296,-0.13833207879751022,-1.6733138018708225>, 1 }        
    sphere {  m*<0.5336723745453924,0.290800793971149,8.285042123638904>, 1 }
    sphere {  m*<2.4829071619296808,-0.035617462703209646,-2.911660000864776>, 1 }
    sphere {  m*<-1.8924924311002345,2.190143694416979,-2.6469707017386166>, 1}
    sphere { m*<-1.6247052100624026,-2.6975482479869184,-2.457424416576046>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5336723745453924,0.290800793971149,8.285042123638904>, <-0.27070624899062296,-0.13833207879751022,-1.6733138018708225>, 0.5 }
    cylinder { m*<2.4829071619296808,-0.035617462703209646,-2.911660000864776>, <-0.27070624899062296,-0.13833207879751022,-1.6733138018708225>, 0.5}
    cylinder { m*<-1.8924924311002345,2.190143694416979,-2.6469707017386166>, <-0.27070624899062296,-0.13833207879751022,-1.6733138018708225>, 0.5 }
    cylinder {  m*<-1.6247052100624026,-2.6975482479869184,-2.457424416576046>, <-0.27070624899062296,-0.13833207879751022,-1.6733138018708225>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    