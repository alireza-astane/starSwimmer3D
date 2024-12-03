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
    sphere { m*<-1.0542701726341128,-0.9737311012150809,-0.7499451887609205>, 1 }        
    sphere {  m*<0.38086722062076284,-0.18024068753232372,9.1147494172882>, 1 }
    sphere {  m*<7.736218658620738,-0.26916096352668,-5.46474387275714>, 1 }
    sphere {  m*<-5.917951079991865,4.94342665530319,-3.239937281193681>, 1}
    sphere { m*<-2.3101281454583855,-3.627028449995125,-1.3673386424738025>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.38086722062076284,-0.18024068753232372,9.1147494172882>, <-1.0542701726341128,-0.9737311012150809,-0.7499451887609205>, 0.5 }
    cylinder { m*<7.736218658620738,-0.26916096352668,-5.46474387275714>, <-1.0542701726341128,-0.9737311012150809,-0.7499451887609205>, 0.5}
    cylinder { m*<-5.917951079991865,4.94342665530319,-3.239937281193681>, <-1.0542701726341128,-0.9737311012150809,-0.7499451887609205>, 0.5 }
    cylinder {  m*<-2.3101281454583855,-3.627028449995125,-1.3673386424738025>, <-1.0542701726341128,-0.9737311012150809,-0.7499451887609205>, 0.5}

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
    sphere { m*<-1.0542701726341128,-0.9737311012150809,-0.7499451887609205>, 1 }        
    sphere {  m*<0.38086722062076284,-0.18024068753232372,9.1147494172882>, 1 }
    sphere {  m*<7.736218658620738,-0.26916096352668,-5.46474387275714>, 1 }
    sphere {  m*<-5.917951079991865,4.94342665530319,-3.239937281193681>, 1}
    sphere { m*<-2.3101281454583855,-3.627028449995125,-1.3673386424738025>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.38086722062076284,-0.18024068753232372,9.1147494172882>, <-1.0542701726341128,-0.9737311012150809,-0.7499451887609205>, 0.5 }
    cylinder { m*<7.736218658620738,-0.26916096352668,-5.46474387275714>, <-1.0542701726341128,-0.9737311012150809,-0.7499451887609205>, 0.5}
    cylinder { m*<-5.917951079991865,4.94342665530319,-3.239937281193681>, <-1.0542701726341128,-0.9737311012150809,-0.7499451887609205>, 0.5 }
    cylinder {  m*<-2.3101281454583855,-3.627028449995125,-1.3673386424738025>, <-1.0542701726341128,-0.9737311012150809,-0.7499451887609205>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    