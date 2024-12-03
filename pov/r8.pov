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
    sphere { m*<1.1165279844606482,-0.045651728985697027,0.5811063807855159>, 1 }        
    sphere {  m*<3.9057443235270997,1.4372953495145497,10.069166274634338>, 1 }
    sphere {  m*<9.06132373007123,0.9052775108699709,-5.416699218465072>, 1 }
    sphere {  m*<-5.802807817527815,6.713970458292578,-1.955084255006736>, 1}
    sphere { m*<-2.6981482982279528,-9.239150234619919,-0.37295727802047207>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<3.9057443235270997,1.4372953495145497,10.069166274634338>, <1.1165279844606482,-0.045651728985697027,0.5811063807855159>, 0.5 }
    cylinder { m*<9.06132373007123,0.9052775108699709,-5.416699218465072>, <1.1165279844606482,-0.045651728985697027,0.5811063807855159>, 0.5}
    cylinder { m*<-5.802807817527815,6.713970458292578,-1.955084255006736>, <1.1165279844606482,-0.045651728985697027,0.5811063807855159>, 0.5 }
    cylinder {  m*<-2.6981482982279528,-9.239150234619919,-0.37295727802047207>, <1.1165279844606482,-0.045651728985697027,0.5811063807855159>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    