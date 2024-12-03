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
    sphere { m*<1.0210997069833532,0.4382449032855257,0.4696093655479694>, 1 }        
    sphere {  m*<1.265039371779842,0.473449191650571,3.459466423999144>, 1 }
    sphere {  m*<3.7582865608423774,0.4734491916505708,-0.7578157844914746>, 1 }
    sphere {  m*<-2.8990833142601438,6.608593699715697,-1.8482549467594798>, 1}
    sphere { m*<-3.8043758633817126,-7.8423776449881615,-2.3828494166920633>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.265039371779842,0.473449191650571,3.459466423999144>, <1.0210997069833532,0.4382449032855257,0.4696093655479694>, 0.5 }
    cylinder { m*<3.7582865608423774,0.4734491916505708,-0.7578157844914746>, <1.0210997069833532,0.4382449032855257,0.4696093655479694>, 0.5}
    cylinder { m*<-2.8990833142601438,6.608593699715697,-1.8482549467594798>, <1.0210997069833532,0.4382449032855257,0.4696093655479694>, 0.5 }
    cylinder {  m*<-3.8043758633817126,-7.8423776449881615,-2.3828494166920633>, <1.0210997069833532,0.4382449032855257,0.4696093655479694>, 0.5}

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
    sphere { m*<1.0210997069833532,0.4382449032855257,0.4696093655479694>, 1 }        
    sphere {  m*<1.265039371779842,0.473449191650571,3.459466423999144>, 1 }
    sphere {  m*<3.7582865608423774,0.4734491916505708,-0.7578157844914746>, 1 }
    sphere {  m*<-2.8990833142601438,6.608593699715697,-1.8482549467594798>, 1}
    sphere { m*<-3.8043758633817126,-7.8423776449881615,-2.3828494166920633>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.265039371779842,0.473449191650571,3.459466423999144>, <1.0210997069833532,0.4382449032855257,0.4696093655479694>, 0.5 }
    cylinder { m*<3.7582865608423774,0.4734491916505708,-0.7578157844914746>, <1.0210997069833532,0.4382449032855257,0.4696093655479694>, 0.5}
    cylinder { m*<-2.8990833142601438,6.608593699715697,-1.8482549467594798>, <1.0210997069833532,0.4382449032855257,0.4696093655479694>, 0.5 }
    cylinder {  m*<-3.8043758633817126,-7.8423776449881615,-2.3828494166920633>, <1.0210997069833532,0.4382449032855257,0.4696093655479694>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    