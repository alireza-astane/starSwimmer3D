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
    sphere { m*<-0.3453873361519261,-0.27272641087752536,-0.40950858096051956>, 1 }        
    sphere {  m*<1.0737801580482338,0.717212503002391,9.439781516074616>, 1 }
    sphere {  m*<8.44156735637102,0.43212025221012884,-5.130895912999301>, 1 }
    sphere {  m*<-6.454395837317962,6.955201625830763,-3.640089009817695>, 1}
    sphere { m*<-4.2790261856553,-8.839423751367747,-2.2311277558877567>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0737801580482338,0.717212503002391,9.439781516074616>, <-0.3453873361519261,-0.27272641087752536,-0.40950858096051956>, 0.5 }
    cylinder { m*<8.44156735637102,0.43212025221012884,-5.130895912999301>, <-0.3453873361519261,-0.27272641087752536,-0.40950858096051956>, 0.5}
    cylinder { m*<-6.454395837317962,6.955201625830763,-3.640089009817695>, <-0.3453873361519261,-0.27272641087752536,-0.40950858096051956>, 0.5 }
    cylinder {  m*<-4.2790261856553,-8.839423751367747,-2.2311277558877567>, <-0.3453873361519261,-0.27272641087752536,-0.40950858096051956>, 0.5}

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
    sphere { m*<-0.3453873361519261,-0.27272641087752536,-0.40950858096051956>, 1 }        
    sphere {  m*<1.0737801580482338,0.717212503002391,9.439781516074616>, 1 }
    sphere {  m*<8.44156735637102,0.43212025221012884,-5.130895912999301>, 1 }
    sphere {  m*<-6.454395837317962,6.955201625830763,-3.640089009817695>, 1}
    sphere { m*<-4.2790261856553,-8.839423751367747,-2.2311277558877567>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0737801580482338,0.717212503002391,9.439781516074616>, <-0.3453873361519261,-0.27272641087752536,-0.40950858096051956>, 0.5 }
    cylinder { m*<8.44156735637102,0.43212025221012884,-5.130895912999301>, <-0.3453873361519261,-0.27272641087752536,-0.40950858096051956>, 0.5}
    cylinder { m*<-6.454395837317962,6.955201625830763,-3.640089009817695>, <-0.3453873361519261,-0.27272641087752536,-0.40950858096051956>, 0.5 }
    cylinder {  m*<-4.2790261856553,-8.839423751367747,-2.2311277558877567>, <-0.3453873361519261,-0.27272641087752536,-0.40950858096051956>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    