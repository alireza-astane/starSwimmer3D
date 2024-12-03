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
    sphere { m*<1.0116988466278554,0.4534410941495094,0.4640509554101393>, 1 }        
    sphere {  m*<1.2556140891219012,0.49001450434276483,3.4538935174408865>, 1 }
    sphere {  m*<3.748861278184437,0.49001450434276467,-0.7633886910497316>, 1 }
    sphere {  m*<-2.869701531941574,6.552638767206747,-1.8308821361255558>, 1}
    sphere { m*<-3.8081399131112357,-7.831611645649728,-2.385075170104762>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2556140891219012,0.49001450434276483,3.4538935174408865>, <1.0116988466278554,0.4534410941495094,0.4640509554101393>, 0.5 }
    cylinder { m*<3.748861278184437,0.49001450434276467,-0.7633886910497316>, <1.0116988466278554,0.4534410941495094,0.4640509554101393>, 0.5}
    cylinder { m*<-2.869701531941574,6.552638767206747,-1.8308821361255558>, <1.0116988466278554,0.4534410941495094,0.4640509554101393>, 0.5 }
    cylinder {  m*<-3.8081399131112357,-7.831611645649728,-2.385075170104762>, <1.0116988466278554,0.4534410941495094,0.4640509554101393>, 0.5}

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
    sphere { m*<1.0116988466278554,0.4534410941495094,0.4640509554101393>, 1 }        
    sphere {  m*<1.2556140891219012,0.49001450434276483,3.4538935174408865>, 1 }
    sphere {  m*<3.748861278184437,0.49001450434276467,-0.7633886910497316>, 1 }
    sphere {  m*<-2.869701531941574,6.552638767206747,-1.8308821361255558>, 1}
    sphere { m*<-3.8081399131112357,-7.831611645649728,-2.385075170104762>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2556140891219012,0.49001450434276483,3.4538935174408865>, <1.0116988466278554,0.4534410941495094,0.4640509554101393>, 0.5 }
    cylinder { m*<3.748861278184437,0.49001450434276467,-0.7633886910497316>, <1.0116988466278554,0.4534410941495094,0.4640509554101393>, 0.5}
    cylinder { m*<-2.869701531941574,6.552638767206747,-1.8308821361255558>, <1.0116988466278554,0.4534410941495094,0.4640509554101393>, 0.5 }
    cylinder {  m*<-3.8081399131112357,-7.831611645649728,-2.385075170104762>, <1.0116988466278554,0.4534410941495094,0.4640509554101393>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    