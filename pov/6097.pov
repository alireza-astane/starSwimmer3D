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
    sphere { m*<-1.5321613629494693,-0.28892831605150365,-0.9952364676768221>, 1 }        
    sphere {  m*<-0.06917604004970301,0.19530092679918354,8.885397903062458>, 1 }
    sphere {  m*<7.286175397950271,0.10638065080482612,-5.694095386982907>, 1 }
    sphere {  m*<-3.624528405844747,2.5496315778561707,-2.0682273760317766>, 1}
    sphere { m*<-2.9135588155881225,-2.863028111151983,-1.6764512874863944>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.06917604004970301,0.19530092679918354,8.885397903062458>, <-1.5321613629494693,-0.28892831605150365,-0.9952364676768221>, 0.5 }
    cylinder { m*<7.286175397950271,0.10638065080482612,-5.694095386982907>, <-1.5321613629494693,-0.28892831605150365,-0.9952364676768221>, 0.5}
    cylinder { m*<-3.624528405844747,2.5496315778561707,-2.0682273760317766>, <-1.5321613629494693,-0.28892831605150365,-0.9952364676768221>, 0.5 }
    cylinder {  m*<-2.9135588155881225,-2.863028111151983,-1.6764512874863944>, <-1.5321613629494693,-0.28892831605150365,-0.9952364676768221>, 0.5}

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
    sphere { m*<-1.5321613629494693,-0.28892831605150365,-0.9952364676768221>, 1 }        
    sphere {  m*<-0.06917604004970301,0.19530092679918354,8.885397903062458>, 1 }
    sphere {  m*<7.286175397950271,0.10638065080482612,-5.694095386982907>, 1 }
    sphere {  m*<-3.624528405844747,2.5496315778561707,-2.0682273760317766>, 1}
    sphere { m*<-2.9135588155881225,-2.863028111151983,-1.6764512874863944>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.06917604004970301,0.19530092679918354,8.885397903062458>, <-1.5321613629494693,-0.28892831605150365,-0.9952364676768221>, 0.5 }
    cylinder { m*<7.286175397950271,0.10638065080482612,-5.694095386982907>, <-1.5321613629494693,-0.28892831605150365,-0.9952364676768221>, 0.5}
    cylinder { m*<-3.624528405844747,2.5496315778561707,-2.0682273760317766>, <-1.5321613629494693,-0.28892831605150365,-0.9952364676768221>, 0.5 }
    cylinder {  m*<-2.9135588155881225,-2.863028111151983,-1.6764512874863944>, <-1.5321613629494693,-0.28892831605150365,-0.9952364676768221>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    