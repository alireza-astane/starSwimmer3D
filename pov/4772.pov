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
    sphere { m*<-0.23727063789522213,-0.12207558491707136,-1.3002611593659146>, 1 }        
    sphere {  m*<0.4376703507655665,0.23878457094116678,7.075846567265729>, 1 }
    sphere {  m*<2.4974377561110352,-0.02004160953069712,-2.5294706848170945>, 1 }
    sphere {  m*<-1.858885997788112,2.2063983595015273,-2.2742069247818812>, 1}
    sphere { m*<-1.5910987767502802,-2.68129358290237,-2.0846606396193086>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4376703507655665,0.23878457094116678,7.075846567265729>, <-0.23727063789522213,-0.12207558491707136,-1.3002611593659146>, 0.5 }
    cylinder { m*<2.4974377561110352,-0.02004160953069712,-2.5294706848170945>, <-0.23727063789522213,-0.12207558491707136,-1.3002611593659146>, 0.5}
    cylinder { m*<-1.858885997788112,2.2063983595015273,-2.2742069247818812>, <-0.23727063789522213,-0.12207558491707136,-1.3002611593659146>, 0.5 }
    cylinder {  m*<-1.5910987767502802,-2.68129358290237,-2.0846606396193086>, <-0.23727063789522213,-0.12207558491707136,-1.3002611593659146>, 0.5}

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
    sphere { m*<-0.23727063789522213,-0.12207558491707136,-1.3002611593659146>, 1 }        
    sphere {  m*<0.4376703507655665,0.23878457094116678,7.075846567265729>, 1 }
    sphere {  m*<2.4974377561110352,-0.02004160953069712,-2.5294706848170945>, 1 }
    sphere {  m*<-1.858885997788112,2.2063983595015273,-2.2742069247818812>, 1}
    sphere { m*<-1.5910987767502802,-2.68129358290237,-2.0846606396193086>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4376703507655665,0.23878457094116678,7.075846567265729>, <-0.23727063789522213,-0.12207558491707136,-1.3002611593659146>, 0.5 }
    cylinder { m*<2.4974377561110352,-0.02004160953069712,-2.5294706848170945>, <-0.23727063789522213,-0.12207558491707136,-1.3002611593659146>, 0.5}
    cylinder { m*<-1.858885997788112,2.2063983595015273,-2.2742069247818812>, <-0.23727063789522213,-0.12207558491707136,-1.3002611593659146>, 0.5 }
    cylinder {  m*<-1.5910987767502802,-2.68129358290237,-2.0846606396193086>, <-0.23727063789522213,-0.12207558491707136,-1.3002611593659146>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    